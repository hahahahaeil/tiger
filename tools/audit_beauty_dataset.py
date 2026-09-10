#!/usr/bin/env python3
"""Audit GRID Beauty TFRecord splits without modifying them.

Run inside the GRID conda environment.  The JSON output is intended to be
checked into the experiment log directory, not into the source repository.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import tensorflow as tf


def value(feature):
    kind = feature.WhichOneof("kind")
    values = getattr(feature, kind).value
    if kind == "bytes_list":
        return [x.decode("utf-8", errors="replace") for x in values]
    return list(values)


def iter_examples(paths):
    # TFRecordDataset handles both record framing and GZIP compression correctly.
    for path in paths:
        ds = tf.data.TFRecordDataset(str(path), compression_type="GZIP")
        for raw in ds:
            example = tf.train.Example()
            example.ParseFromString(bytes(raw.numpy()))
            yield {name: value(feat) for name, feat in example.features.feature.items()}


def file_manifest(root):
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            lines.append(f"{path.stat().st_size} {path.relative_to(root).as_posix()}")
    digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    return {"sha256_size_path_manifest": digest, "file_count": len(lines), "total_bytes": sum(int(x.split(" ", 1)[0]) for x in lines)}


def scalar_int(row, name):
    values = row.get(name, [])
    return int(values[0]) if values else None


def audit_items(paths):
    ids, duplicate_rows, rows = set(), 0, 0
    for row in iter_examples(paths):
        rows += 1
        item_id = scalar_int(row, "id")
        if item_id in ids:
            duplicate_rows += 1
        else:
            ids.add(item_id)
    ordered = sorted(ids)
    minimum, maximum = (ordered[0], ordered[-1]) if ordered else (None, None)
    expected = set(range(minimum, maximum + 1)) if ordered else set()
    return {
        "records": rows, "unique_item_ids": len(ids), "min_item_id": minimum,
        "max_item_id": maximum, "duplicate_item_rows": duplicate_rows,
        "missing_ids_within_min_max": len(expected - ids),
        "id_domain_is_zero_based_contiguous": ordered == list(range(len(ordered))),
        "id_domain_is_one_based_contiguous": ordered == list(range(1, len(ordered) + 1)),
        "item_ids": ids,
    }


def audit_split(name, paths, item_ids):
    lengths, all_ids, unknown, rows = Counter(), Counter(), 0, 0
    users, duplicate_user_rows, user_sequences = set(), 0, {}
    last_items = Counter()
    for row in iter_examples(paths):
        rows += 1
        seq = [int(x) for x in row.get("sequence_data", [])]
        lengths[len(seq)] += 1
        all_ids.update(seq)
        unknown += sum(item not in item_ids for item in seq)
        if seq:
            last_items[seq[-1]] += 1
        user = scalar_int(row, "user_id")
        if user is not None:
            if user in users:
                duplicate_user_rows += 1
            users.add(user)
            user_sequences[user] = seq
    return {
        "split": name, "records": rows, "unique_users": len(users),
        "duplicate_user_rows": duplicate_user_rows, "sequence_length_histogram": dict(sorted(lengths.items())),
        "sequence_item_min": min(all_ids) if all_ids else None,
        "sequence_item_max": max(all_ids) if all_ids else None,
        "unknown_item_references": unknown, "unique_sequence_items": len(all_ids),
        "last_item_unique_count": len(last_items),
    }, user_sequences


def audit_split_relationships(training, evaluation, testing):
    users = {name: set(rows) for name, rows in {
        "training": training, "evaluation": evaluation, "testing": testing
    }.items()}
    common = set.intersection(*users.values())
    test_prefix_matches = sum(evaluation[u] == testing[u][:-1] for u in common)
    train_tail_matches = sum(
        len(evaluation[u]) >= 2 and training[u][-len(evaluation[u][:-1]):] == evaluation[u][:-1]
        for u in common
    )
    return {
        "common_users": len(common),
        "users_missing_from_any_split": sum(len(s - common) for s in users.values()),
        "evaluation_equals_testing_without_final_item": test_prefix_matches,
        "training_tail_equals_evaluation_without_final_item": train_tail_matches,
        "all_users_follow_test_holdout_rule": test_prefix_matches == len(common),
        "all_users_follow_validation_holdout_rule": train_tail_matches == len(common),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, help="Beauty directory containing items/training/evaluation/testing")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    required = ("items", "training", "evaluation", "testing")
    missing = [name for name in required if not (args.data_dir / name).is_dir()]
    if missing:
        raise SystemExit(f"Missing required directories: {missing}")
    files = {name: sorted((args.data_dir / name).glob("*.tfrecord.gz")) for name in required}
    item_audit = audit_items(files["items"])
    split_results = {}
    split_sequences = {}
    for name in ("training", "evaluation", "testing"):
        split_results[name], split_sequences[name] = audit_split(name, files[name], item_audit["item_ids"])
    result = {
        "data_dir": str(args.data_dir.resolve()), "manifest": file_manifest(args.data_dir),
        "file_counts": {name: len(paths) for name, paths in files.items()},
        "items": {k: v for k, v in item_audit.items() if k != "item_ids"},
        "splits": split_results,
        "split_relationships": audit_split_relationships(**split_sequences),
        "interpretation": "GRID masks the final num_hierarchies SID tokens in collate. Raw split records therefore contain targets by design; runtime audit must verify that masked labels are absent from model inputs.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
