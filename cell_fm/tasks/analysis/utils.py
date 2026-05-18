import numpy as np
import pandas as pd

def moving_average(y, window: int):
    y = np.asarray(y, dtype=float)
    window = int(max(1, min(window, len(y))))
    return (
        pd.Series(y)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )

def integral(xs: np.ndarray, ys: np.ndarray) -> float:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    if xs.ndim != 1 or ys.ndim != 1:
        raise ValueError("xs and ys must be 1D arrays")
    if xs.shape[0] != ys.shape[0]:
        raise ValueError("xs and ys must have the same length")
    if xs.shape[0] < 2:
        return 0.0

    return float(np.trapz(ys, xs))


def compute_concentration(df: pd.DataFrame, window_size: int, threshold_prob: float = 0.8) -> float:
    intensity_levels = df['protein_intensity_level'].to_numpy(dtype=float)
    predicted_classes = df['predicted_class'].to_numpy(dtype=float)

    predicted_classes_ma = moving_average(predicted_classes, window_size)

    # find the first index where predicted_classes_ma exceeds threshold_prob and return as the concentration.
    for i in range(len(predicted_classes_ma)):
        if predicted_classes_ma[i] >= threshold_prob:
            return intensity_levels[i]

    return float('inf')  # if never exceeds the threshold, return infinity as the concentration (i.e., no condensate formation even at the highest intensity level)


def compute_area_under_curve(df: pd.DataFrame, window_size: int, normalize: bool = True, use_log_scale: bool = True) -> float:
    intensity_levels = df['protein_intensity_level'].to_numpy(dtype=float)
    predicted_classes = df['predicted_class'].to_numpy(dtype=float)

    predicted_classes_ma = moving_average(predicted_classes, window_size)

    if use_log_scale:
        # Integrate against log-scaled intensity.
        intensity_levels_log = np.log10(np.clip(intensity_levels, a_min=1.0, a_max=None))

        # Ensure x-axis is monotonic for stable trapezoidal integration.
        order = np.argsort(intensity_levels_log)
        intensity_levels_log = intensity_levels_log[order]
        predicted_classes_ma = predicted_classes_ma[order]

        area = integral(intensity_levels_log, predicted_classes_ma)
    else:
        # Ensure x-axis is monotonic for stable trapezoidal integration.
        order = np.argsort(intensity_levels)
        intensity_levels = intensity_levels[order]
        predicted_classes_ma = predicted_classes_ma[order]
        area = integral(intensity_levels, predicted_classes_ma)

    if normalize:
        if use_log_scale:
            total_area = (intensity_levels_log[-1] - intensity_levels_log[0]) * 1.0
        else:
            total_area = (intensity_levels[-1] - intensity_levels[0]) * 1.0
        area /= total_area

    return area


def compute_area_if_no_reentrant(df: pd.DataFrame, window_size: int, normalize: bool = True, use_log_scale: bool = True) -> float:
    intensity_levels = df['protein_intensity_level'].to_numpy(dtype=float)
    predicted_classes = df['predicted_class'].to_numpy(dtype=float)

    predicted_classes_ma = moving_average(predicted_classes, window_size)

    # find the max of predicted_classes_ma
    max_pred = np.max(predicted_classes_ma)

    # find the index of the max value
    max_index = np.argmax(predicted_classes_ma)

    no_reentrant_curve = predicted_classes_ma.copy()
    no_reentrant_curve[max_index:] = max_pred

    if use_log_scale:
        # Integrate against log-scaled intensity.
        intensity_levels_log = np.log10(np.clip(intensity_levels, a_min=1.0, a_max=None))

        # Ensure x-axis is monotonic for stable trapezoidal integration.
        order = np.argsort(intensity_levels_log)
        intensity_levels_log = intensity_levels_log[order]
        no_reentrant_curve = no_reentrant_curve[order]

        area = integral(intensity_levels_log, no_reentrant_curve)
    else:
        # Ensure x-axis is monotonic for stable trapezoidal integration.
        order = np.argsort(intensity_levels)
        intensity_levels = intensity_levels[order]
        no_reentrant_curve = no_reentrant_curve[order]
        area = integral(intensity_levels, no_reentrant_curve)

    if normalize:
        if use_log_scale:
            total_area = (intensity_levels_log[-1] - intensity_levels_log[0]) * 1.0
        else:
            total_area = (intensity_levels[-1] - intensity_levels[0]) * 1.0
        area /= total_area

    return area


def compute_area_if_no_reentrant_minus_auc(df: pd.DataFrame, window_size: int, normalize: bool = True, use_log_scale: bool = True) -> float:
    intensity_levels = df['protein_intensity_level'].to_numpy(dtype=float)
    predicted_classes = df['predicted_class'].to_numpy(dtype=float)

    predicted_classes_ma = moving_average(predicted_classes, window_size)

    # find the max of predicted_classes_ma
    max_pred = np.max(predicted_classes_ma)

    # find the index of the max value
    max_index = np.argmax(predicted_classes_ma)

    no_reentrant_curve = predicted_classes_ma.copy()
    no_reentrant_curve[max_index:] = max_pred

    if use_log_scale:
        # Integrate against log-scaled intensity.
        intensity_levels_log = np.log10(np.clip(intensity_levels, a_min=1.0, a_max=None))

        # Ensure x-axis is monotonic for stable trapezoidal integration.
        order = np.argsort(intensity_levels_log)
        intensity_levels_log = intensity_levels_log[order]
        no_reentrant_curve = no_reentrant_curve[order]

        area = integral(intensity_levels_log, no_reentrant_curve - predicted_classes_ma)
    else:
        # Ensure x-axis is monotonic for stable trapezoidal integration.
        order = np.argsort(intensity_levels)
        intensity_levels = intensity_levels[order]
        no_reentrant_curve = no_reentrant_curve[order]
        area = integral(intensity_levels, no_reentrant_curve - predicted_classes_ma)

    if normalize:
        if use_log_scale:
            total_area = (intensity_levels_log[-1] - intensity_levels_log[0]) * 1.0
        else:
            total_area = (intensity_levels[-1] - intensity_levels[0]) * 1.0
        area /= total_area

    return area