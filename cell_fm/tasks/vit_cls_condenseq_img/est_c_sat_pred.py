import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass, asdict
import pandas as pd
from glob import glob
from tqdm import tqdm
import json
from pathlib import Path


def _np_convert(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

# =========================
# Utility: weighted BCE loss and BIC
# =========================

def weighted_bce_loss(q, X, w, eps=1e-8):
    """
    q: predicted prob of X=1, shape [N]
    X: observed {0,1}, shape [N]
    w: weights >=0, shape [N]
    returns scalar loss (mean weighted neg log likelihood)
    """
    loss_per_point = -(X * torch.log(q + eps) + (1 - X) * torch.log(1 - q + eps))
    return (w * loss_per_point).sum() / (w.sum() + eps)

def weighted_log_likelihood(q, X, w, eps=1e-8):
    """
    returns total weighted log-likelihood (note: not averaged)
    """
    ll_per_point = (X * torch.log(q + eps) + (1 - X) * torch.log(1 - q + eps))
    return (w * ll_per_point).sum()


def compute_bic(log_like, k_params, w):
    """
    BIC = -2 * logL + k * log(n_eff)
    where n_eff = sum_i w_i
    log_like: weighted log-likelihood (scalar torch)
    k_params: number of free parameters in the model
    w: weights tensor
    """
    with torch.no_grad():
        n_eff = w.sum().item()
        bic = -2.0 * log_like.item() + k_params * np.log(max(n_eff, 1e-8))
    return bic


# =========================
# Model A: Null model
# q(C) = alpha (constant false-positive rate)
# params: alpha in (0,1)
# =========================

class NullModelClosedForm(nn.Module):
    """
    Null model: q(C) = alpha
    """
    def __init__(self, alpha_hat: float, device=None, dtype=torch.float32):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(float(alpha_hat), dtype=dtype, device=device))

    def forward_q(self, C):
        q = self.alpha.expand_as(C)
        return q, self.alpha

    def num_params(self):
        return 1


def fit_null_model(C, X, w, eps=1e-8):
    w_sum = w.sum().item()
    if w_sum <= 0:
        raise ValueError("Sum of weights must be positive for the null model closed-form MLE.")

    num = (w * X).sum().item()
    alpha_hat = num / w_sum

    alpha_hat = float(np.clip(alpha_hat, eps, 1.0 - eps))

    model = NullModelClosedForm(alpha_hat, device=C.device, dtype=C.dtype)

    with torch.no_grad():
        q, alpha = model.forward_q(C)
        ll = weighted_log_likelihood(q, X, w)  # total weighted log-likelihood (not averaged)
    bic = compute_bic(ll, model.num_params(), w)

    results = {
        "alpha": alpha_hat,
        "log_likelihood": ll.item(),
        "bic": bic,
    }
    return model, results

# =========================
# Model B: Phase-separation model
#
# q(C) = alpha + (beta - alpha) * sigmoid(k * (C - C_sat))
#
# params:
#   C_sat : real (unconstrained)
#   k     : >0      -> softplus(raw_k)
#   alpha : (0,1)   -> sigmoid(raw_alpha)
#   beta  : (0,1)   -> sigmoid(raw_beta)
#
# =========================

class PhaseSepModel(nn.Module):
    def __init__(self,
                 init_C_sat=None,
                 init_k=1.0,
                 init_alpha=0.05,
                 init_beta=0.95):
        super().__init__()

        if init_C_sat is None:
            init_C_sat = 0.0

        # raw params
        self.C_sat = nn.Parameter(torch.tensor([init_C_sat], dtype=torch.float32))  # unconstrained
        self.raw_k = nn.Parameter(torch.tensor([np.log(np.exp(init_k)-1)], dtype=torch.float32))  # softplus^-1
        self.raw_alpha = nn.Parameter(torch.tensor([np.log(init_alpha/(1-init_alpha))], dtype=torch.float32))
        self.raw_beta  = nn.Parameter(torch.tensor([np.log(init_beta/(1-init_beta))], dtype=torch.float32))

    def forward_q(self, C):
        """
        returns:
            q(C): observed prob of calling condensate
            p(C): latent true condensate prob before measurement noise
            (k_pos, alpha01, beta01)
        """
        k_pos   = torch.nn.functional.softplus(self.raw_k)           # >0
        alpha01 = torch.sigmoid(self.raw_alpha)                      # in (0,1)
        beta01  = torch.sigmoid(self.raw_beta)                       # in (0,1)

        # latent probability of real condensate
        p = torch.sigmoid(k_pos * (C - self.C_sat))

        # observed probability after FP/FN noise
        q = alpha01 + (beta01 - alpha01) * p
        return q, p, (k_pos, alpha01, beta01)

    def num_params(self):
        return 4


def fit_phase_sep_model(C, X, w, max_steps=20000, lr=1e-2):
    # good init for C_sat might be median of concentrations where X==1 (if any),
    # else overall median
    with torch.no_grad():
        if (X > 0.5).any():
            init_C_sat_val = torch.median(C[X > 0.5]).item()
        else:
            init_C_sat_val = torch.median(C).item()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = PhaseSepModel(init_C_sat=init_C_sat_val)
    opt = optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)
    C = C.to(device)
    X = X.to(device)
    w = w.to(device)

    for step in range(max_steps):
        opt.zero_grad()
        q, p, (k_pos, alpha01, beta01) = model.forward_q(C)
        loss = weighted_bce_loss(q, X, w)
        loss.backward()
        opt.step()

    with torch.no_grad():
        q, p, (k_pos, alpha01, beta01) = model.forward_q(C)
        ll = weighted_log_likelihood(q, X, w)

    bic = compute_bic(ll, model.num_params(), w)

    results = {
        "C_sat": model.C_sat.item(),
        "k": k_pos.item(),
        "alpha": alpha01.item(),
        "beta": beta01.item(),
        "log_likelihood": ll.item(),
        "bic": bic,
    }

    return model, results


# =========================
# High-level decision logic
# =========================

@dataclass
class CondensateDecision:
    condenses: bool
    model_choice: str  # "null" or "phase_sep"
    C_sat: float | None
    C_sat_lower_bound: float | None
    k: float | None
    alpha: float
    beta: float | None
    bic_null: float
    bic_phase: float
    log_like_null: float
    log_like_phase: float
    extra_info: dict


def bootstrap_Csat(C, X, w, n_boot=200, max_steps=2000, lr=1e-2):
    """
    Optional: bootstrap uncertainty on C_sat for the phase-separation model.
    We'll resample indices with replacement (weighted by w as sampling weight).
    Returns list of C_sat estimates. If phase_sep fit collapses, skip that sample.
    """
    N = len(C)
    # sampling probabilities proportional to w to emphasize high-confidence points
    probs = (w / (w.sum() + 1e-8)).cpu().numpy()

    C_sat_list = []
    for b in range(n_boot):
        idx = np.random.choice(np.arange(N), size=N, replace=True, p=probs)
        C_b = C[idx]
        X_b = X[idx]
        w_b = w[idx]

        # fit model B on bootstrap sample
        try:
            _, res_b = fit_phase_sep_model(C_b, X_b, w_b, max_steps=max_steps, lr=lr)
            C_sat_list.append(res_b["C_sat"])
        except Exception:
            # if optimizer blew up / degenerate, just skip
            pass

    if len(C_sat_list) == 0:
        return None, None, None

    arr = np.array(C_sat_list)
    lower = np.percentile(arr, 2.5)
    upper = np.percentile(arr, 97.5)
    median = np.median(arr)
    return median, lower, upper


def analyze_condensation(
    C_input,
    X_input,
    w_input=None,
    bic_delta_threshold=0,
    seed=0,
):
    """
    Main entry point.

    C_input: array-like of concentrations [N]
    X_input: array-like of 0/1 calls (did we see condensate?) [N]
    w_input: array-like of weights/confidence [N] or None (-> all ones)

    Returns: CondensateDecision dataclass
    """

    # ---- prep data
    rng = np.random.default_rng(seed)

    C_arr = np.asarray(C_input, dtype=np.float32).reshape(-1)
    X_arr = np.asarray(X_input, dtype=np.float32).reshape(-1)
    assert C_arr.shape == X_arr.shape, "C and X must have same length."

    if w_input is None:
        w_arr = np.ones_like(C_arr, dtype=np.float32)
    else:
        w_arr = np.asarray(w_input, dtype=np.float32).reshape(-1)
        assert w_arr.shape == C_arr.shape, "w must have same length as C and X."

    # convert to torch tensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    C = torch.tensor(C_arr, dtype=torch.float32, device=device)
    X = torch.tensor(X_arr, dtype=torch.float32, device=device)
    w = torch.tensor(w_arr, dtype=torch.float32, device=device)

    # ---- fit both models
    null_model, null_res = fit_null_model(C, X, w)
    phase_model, phase_res = fit_phase_sep_model(C, X, w)

    bic_null = null_res["bic"]
    bic_phase = phase_res["bic"]

    # ---- decide which model wins
    # if null clearly better (by bic_delta_threshold), call it non-condensing
    # else call phase-separating
    if bic_null + bic_delta_threshold < bic_phase or np.all(X_arr == 0):
        # choose null model
        condenses = False
        model_choice = "null"

        # define a lower bound on C_sat:
        # we say C_sat > high_conc, where high_conc ~ top 5% of observed C
        high_cut = np.percentile(C_arr, 95.0)
        C_sat_lower_bound = float(high_cut)

        # fill result fields
        decision = CondensateDecision(
            condenses=False,
            model_choice=model_choice,
            C_sat=None,
            C_sat_lower_bound=C_sat_lower_bound,
            k=None,
            alpha=null_res["alpha"],  # background FP rate
            beta=None,
            bic_null=bic_null,
            bic_phase=bic_phase,
            log_like_null=null_res["log_likelihood"],
            log_like_phase=phase_res["log_likelihood"],
            extra_info={
                "note": "No significant concentration-dependent increase in condensate calls. "
                        "Treat as non-condensing in tested range.",
            },
        )
        return decision
    else:
        # choose phase-separation model
        condenses = True
        model_choice = "phase_sep"

        decision = CondensateDecision(
            condenses=True,
            model_choice=model_choice,
            C_sat=phase_res["C_sat"],
            C_sat_lower_bound=None,
            k=phase_res["k"],
            alpha=phase_res["alpha"],
            beta=phase_res["beta"],
            bic_null=bic_null,
            bic_phase=bic_phase,
            log_like_null=null_res["log_likelihood"],
            log_like_phase=phase_res["log_likelihood"],
            extra_info={
                "note": "Phase-separation model supported over null model.",
            },
        )
        return decision


# =========================
# Example usage
# =========================

if __name__ == "__main__":
    pred_name = 'PT_CondenSeq_CELL-Diff2_Rep_Dev_GFP_e_4_ignl_4_ignah_8_S1_R1_50k'
    all_data_files = sorted(glob(f"/hpc/reference/opencell/condenseq/seq2img_exp_level/{pred_name}/cellfm_test/*/pred_vs_intensity.csv"))

    for data_file in tqdm(all_data_files):
        output_dir = "/".join(data_file.split("/")[:-1])

        data = pd.read_csv(data_file)

        C_real = data["protein_intensity_level"].values
        X_real = data["predicted_class"].values

        C_real = np.array(C_real, dtype=np.float32)
        X_real = np.array(X_real, dtype=np.float32)

        decision_real = analyze_condensation(C_real, X_real)

        print("\n=== DECISION (real data) ===")
        for k, v in asdict(decision_real).items():
            print(f"{k}: {v}")
        
        # save decision to json
        decision_path = Path(output_dir) / "decision_real.json"
        with open(decision_path, "w", encoding="utf-8") as f:
            json.dump(asdict(decision_real), f, ensure_ascii=False, indent=2, default=_np_convert)
        print(f"Saved decision to {decision_path}")

        # plot data and fitted curves
        import matplotlib.pyplot as plt

        if decision_real.condenses:
            # --- gather fitted params ---
            C_sat = decision_real.C_sat
            k_hat = decision_real.k
            alpha_hat = decision_real.alpha
            beta_hat = decision_real.beta

            # --- compute fitted q(C) over a nice x-grid ---
            x_min = float(np.min(C_real))
            x_max = float(np.max(C_real))
            x_pad = 0.05 * (x_max - x_min + 1e-8)
            xs = np.linspace(x_min - x_pad, x_max + x_pad, 400).astype(np.float32)

            def q_fit(c):
                return alpha_hat + (beta_hat - alpha_hat) * (1.0 / (1.0 + np.exp(-k_hat * (c - C_sat))))

            q_curve = q_fit(xs)

            # --- empirical binning for visibility ---
            # 按浓度分箱，画每个bin的加权阳性率（这里没有单点权重，就用等权；如果有 w 可以在上游改）
            num_bins = 12
            bins = np.linspace(x_min, x_max, num_bins + 1)
            bin_ids = np.digitize(C_real, bins) - 1  # 0..num_bins-1
            bin_x = []
            bin_rate = []
            for b in range(num_bins):
                m = bin_ids == b
                if m.any():
                    bin_x.append(0.5 * (bins[b] + bins[b + 1]))
                    bin_rate.append(X_real[m].mean())
            bin_x = np.array(bin_x, dtype=np.float32)
            bin_rate = np.array(bin_rate, dtype=np.float32)

            # --- scatter of raw points with jitter on Y to reduce overlap ---
            rng = np.random.default_rng(0)
            y_jitter = (rng.random(len(X_real)) - 0.5) * 0.08  # ±0.04
            y_scatter = X_real + y_jitter

            plt.figure(figsize=(7, 4.5), dpi=140)
            # raw points
            plt.scatter(C_real, y_scatter, s=12, alpha=0.35, label="single cells (jittered)")
            # binned empirical rate
            if len(bin_x) > 0:
                plt.plot(bin_x, bin_rate, marker="o", linestyle="none", markersize=6,
                        label="empirical rate (binned)")
            # fitted curve
            plt.plot(xs, q_curve, linewidth=2, label="fitted q(C)")

            # C_sat vertical line
            plt.axvline(C_sat, linestyle="--", linewidth=1.5, label=f"C_sat = {C_sat:.3f}")

            plt.ylim(-0.15, 1.15)
            plt.xlabel("Concentration")
            plt.ylabel("P(condensate)")
            plt.title("Condensate probability vs. concentration")
            plt.legend(loc="best", frameon=False)
            plt.tight_layout()
            plt.savefig(Path(output_dir) / "condensation_fit_plot.png", dpi=300)
            plt.close()
        else:
            print("Model decision: non-condensing within tested range; skip fit plot.")