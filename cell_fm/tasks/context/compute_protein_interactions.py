"""
Compute pairwise interaction embeddings between query proteins and context proteins.

For each (query, context) pair, the interaction is represented by embedding the
concatenated sequence: query_seq + GS_linker + context_seq through ESMC 600m,
then average-pooling to a single 1152-d vector.

Output: per-shard HDF5 files with shape [n_query, n_context, 1152].
"""

import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd
import torch
from esm.models.esmc import ESMC
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.utils import encoding
from tqdm import tqdm

GS_LINKER = "GGGGS" * 3  # 15 aa flexible linker
ESM_DIM = 1152


def parse_args():
    parser = argparse.ArgumentParser(description="Compute protein interaction embeddings via ESMC 600m")
    parser.add_argument("--query_csv", required=True, help="Path to all_merged_meta_data.csv")
    parser.add_argument("--context_csv", required=True, help="Path to HEK293T_500_candidate_proteins_sequences.csv")
    parser.add_argument("--output_dir", required=True, help="Directory to save HDF5 shards and skip logs")
    parser.add_argument("--shard_id", type=int, default=0, help="0-indexed shard index")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--batch_size", type=int, default=8, help="Context proteins per batch")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Max combined token length; pairs above this are skipped")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_data(query_csv, context_csv):
    df_query = pd.read_csv(query_csv)
    df_context = pd.read_csv(context_csv)
    return df_query, df_context


def build_model(device):
    model = ESMC.from_pretrained("esmc_600m").eval()
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model


def tokenize(seq, vocab):
    return encoding.tokenize_sequence(seq, vocab, True)


def embed_batch(token_list, model, vocab, device):
    """Embed a list of token tensors (already filtered for length). Returns [B, 1152]."""
    pad_id = vocab.pad_token_id
    max_len = max(t.size(0) for t in token_list)
    batch = torch.full((len(token_list), max_len), pad_id, dtype=torch.long, device=device)
    for i, t in enumerate(token_list):
        batch[i, :t.size(0)] = t.to(device)

    with torch.no_grad():
        embeddings = model(batch).embeddings  # [B, seq_len, 1152]

    embeddings = embeddings.to(torch.float32)

    # Build padding mask to exclude pad tokens from average pool
    pad_mask = (batch != pad_id).unsqueeze(-1).float()  # [B, seq_len, 1]
    pooled = (embeddings * pad_mask).sum(dim=1) / pad_mask.sum(dim=1)  # [B, 1152]
    return pooled.cpu().numpy()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df_query, df_context = load_data(args.query_csv, args.context_csv)

    # Round-robin sharding for balanced sequence length distribution
    all_query = df_query["gene_name"].tolist()
    shard_indices = list(range(args.shard_id, len(all_query), args.num_shards))
    shard_df = df_query.iloc[shard_indices].reset_index(drop=True)

    context_gene_names = df_context["gene_name"].tolist()
    context_seqs = df_context["sequence"].tolist()
    n_context = len(context_seqs)
    n_query = len(shard_df)

    print(f"Shard {args.shard_id}/{args.num_shards}: {n_query} query proteins x {n_context} context proteins")

    vocab = EsmSequenceTokenizer()
    model = build_model(args.device)

    # Pre-tokenize context sequences and filter those exceeding max_seq_len individually
    context_tokens = []
    valid_context_indices = []
    skipped_context = []
    for ci, (ctx_seq, ctx_gene) in enumerate(zip(context_seqs, context_gene_names)):
        tokens = tokenize(ctx_seq, vocab)
        if tokens.size(0) >= args.max_seq_len:
            skipped_context.append(ci)
        else:
            valid_context_indices.append(ci)
            context_tokens.append(tokens)

    print(f"Context proteins: {len(valid_context_indices)} valid, {len(skipped_context)} skipped (seq len >= {args.max_seq_len})")

    out_h5 = os.path.join(args.output_dir, f"shard_{args.shard_id:04d}_of_{args.num_shards:04d}.h5")
    out_skip = os.path.join(args.output_dir, f"shard_{args.shard_id:04d}_of_{args.num_shards:04d}_skipped.jsonl")

    # Allocate output arrays
    embeddings = np.zeros((n_query, n_context, ESM_DIM), dtype=np.float32)
    valid_mask = np.zeros((n_query, n_context), dtype=bool)

    with open(out_skip, "w") as skip_file:
        # Log skipped context proteins upfront
        for ci in skipped_context:
            skip_file.write(json.dumps({
                "reason": "context_seq_too_long",
                "context": context_gene_names[ci],
            }) + "\n")

        for qi, row in enumerate(tqdm(shard_df.itertuples(), total=n_query,
                                      desc=f"Shard {args.shard_id}")):
            query_seq = row.sequence
            query_gene = row.gene_name

            # Skip query proteins whose individual sequence exceeds max_seq_len
            query_tokens_check = tokenize(query_seq, vocab)
            if query_tokens_check.size(0) >= args.max_seq_len:
                skip_file.write(json.dumps({
                    "reason": "query_seq_too_long",
                    "query": query_gene,
                    "seq_len": query_tokens_check.size(0),
                }) + "\n")
                continue

            # Build combined tokens for each valid context protein
            valid_indices = []
            valid_tokens = []
            for ci, ctx_tokens in zip(valid_context_indices, context_tokens):
                combined = query_seq + GS_LINKER + context_seqs[ci]
                tokens = tokenize(combined, vocab)
                valid_indices.append(ci)
                valid_tokens.append(tokens)

            # Embed in mini-batches
            for batch_start in range(0, len(valid_tokens), args.batch_size):
                batch_tokens = valid_tokens[batch_start:batch_start + args.batch_size]
                batch_indices = valid_indices[batch_start:batch_start + args.batch_size]
                pooled = embed_batch(batch_tokens, model, vocab, args.device)
                for local_i, ci in enumerate(batch_indices):
                    embeddings[qi, ci] = pooled[local_i]
                    valid_mask[qi, ci] = True

    # Save HDF5
    query_gene_names = shard_df["gene_name"].tolist()
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("embeddings", data=embeddings, compression="gzip", compression_opts=4)
        f.create_dataset("valid_mask", data=valid_mask, compression="gzip", compression_opts=4)
        f.create_dataset("query_gene_names", data=np.array(query_gene_names, dtype=h5py.string_dtype()))
        f.create_dataset("context_gene_names", data=np.array(context_gene_names, dtype=h5py.string_dtype()))

    print(f"Saved {out_h5}")
    n_skipped = int((~valid_mask).sum())
    print(f"Skipped {n_skipped} / {n_query * n_context} pairs (logged to {out_skip})")


if __name__ == "__main__":
    main()
