#!/usr/bin/env python3
"""Audit an embedding or post-processed TIGER Semantic-ID tensor."""

import argparse
import json
import sys
from pathlib import Path

import torch


def load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(value).__name__}")
    return value


def audit_embedding(tensor: torch.Tensor, expected_dim: int) -> dict:
    finite = torch.isfinite(tensor)
    return {
        "kind": "embedding",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "item_count": int(tensor.shape[0]) if tensor.ndim else 0,
        "embedding_dim": int(tensor.shape[1]) if tensor.ndim == 2 else None,
        "expected_embedding_dim": expected_dim,
        "dimension_matches": tensor.ndim == 2 and tensor.shape[1] == expected_dim,
        "finite": bool(finite.all().item()),
        "nan_count": int(torch.isnan(tensor).sum().item()),
        "inf_count": int(torch.isinf(tensor).sum().item()),
        "min": float(tensor.min().item()) if tensor.numel() else None,
        "max": float(tensor.max().item()) if tensor.numel() else None,
        "mean": float(tensor.float().mean().item()) if tensor.numel() else None,
    }


def audit_sid(tensor: torch.Tensor, raw_hierarchies: int, codebook_width: int, vocab_size: int) -> dict:
    if tensor.ndim != 2:
        raise ValueError(f"SID tensor must be 2-D, got shape {tuple(tensor.shape)}")
    expected_k = raw_hierarchies + 1
    is_k_by_n = tensor.shape[0] == expected_k
    rows = tensor.t() if is_k_by_n else tensor
    if rows.shape[1] != expected_k:
        raise ValueError(
            f"Expected {expected_k} SID columns (three RQ codes plus collision digit), got {tuple(tensor.shape)}"
        )
    raw = rows[:, :raw_hierarchies]
    collision_digit = rows[:, raw_hierarchies]
    unique_raw, raw_counts = torch.unique(raw, dim=0, return_counts=True)
    unique_final = torch.unique(rows, dim=0).shape[0]
    return {
        "kind": "semantic_id",
        "stored_shape": list(tensor.shape),
        "stored_orientation": "KxN" if is_k_by_n else "NxK (unexpected after export)",
        "item_count": int(rows.shape[0]),
        "sid_width": int(rows.shape[1]),
        "expected_sid_width": expected_k,
        "raw_code_range": [int(raw.min().item()), int(raw.max().item())] if raw.numel() else None,
        "raw_codes_within_codebook": bool(((raw >= 0) & (raw < codebook_width)).all().item()),
        "collision_digit_range": [int(collision_digit.min().item()), int(collision_digit.max().item())] if collision_digit.numel() else None,
        "fourth_digit_generated": bool(torch.any(collision_digit != 0).item()),
        "raw_sid_collision_groups": int((raw_counts > 1).sum().item()),
        "raw_sid_collision_items": int(raw_counts[raw_counts > 1].sum().item()),
        "final_unique_sid_count": int(unique_final),
        "final_sid_collision_count": int(rows.shape[0] - unique_final),
        "reversible_item_to_sid_to_item": bool(unique_final == rows.shape[0]),
        "all_sid_tokens_within_tiger_vocab": bool(((rows >= 0) & (rows < vocab_size)).all().item()),
        "tiger_vocab_size": vocab_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("embedding", "sid"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-dim", type=int, default=2048)
    parser.add_argument("--raw-hierarchies", type=int, default=3)
    parser.add_argument("--codebook-width", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tensor = load_tensor(args.path)
    report = audit_embedding(tensor, args.expected_dim) if args.kind == "embedding" else audit_sid(
        tensor, args.raw_hierarchies, args.codebook_width, args.vocab_size
    )
    report["path"] = str(args.path)
    report["pass"] = bool(
        report["finite"] and report["dimension_matches"]
        if args.kind == "embedding"
        else report["stored_orientation"] == "KxN"
        and report["raw_codes_within_codebook"]
        and report["final_sid_collision_count"] == 0
        and report["reversible_item_to_sid_to_item"]
        and report["all_sid_tokens_within_tiger_vocab"]
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
