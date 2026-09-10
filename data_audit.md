# Beauty data audit

Status: passed on the server `grid` environment. The record-level inspection
used TensorFlow's GZIP TFRecord reader and examined every record.

The actual local Beauty directory is
`D:/data/amazon_data/amazon_data/beauty`, one level below the supplied root.
The actual server counterpart after the observed unzip command is
`/data2/hnlcm/dataset/amazon_data/amazon_data/beauty`. This extra directory
level is recorded here because the original supplied path omitted it.

| Split | Files | Bytes |
| --- | ---: | ---: |
| items | 24 | 35,963,432 |
| training | 175 | 417,282,085 |
| evaluation | 175 | 435,615,164 |
| testing | 175 | 496,831,085 |
| total | 550 | 1,385,699,962 |

The server SHA-256 of the deterministic `size + relative-path` manifest is
`5f8aacfe624a4d32300e67e350168a2ea087f298d84f2c340ac02e5023f7d849`.
This is a layout fingerprint rather than a content hash. File counts and total
bytes match the local filesystem audit; the earlier local manifest used a
different path representation and should not be compared byte-for-byte.

The four required directories exist locally and on the server: `items`,
`training`, `evaluation`, and `testing`. `items` contains 12,101 records with
unique, contiguous, zero-based IDs `0..12100`; no IDs are missing or
duplicated. Each interaction split contains 22,363 users, with no duplicate
user rows and no unknown item references.

Every user follows the intended split protocol. For 21,344 users,
`evaluation == testing[:-1]`; for the remaining 1,019 fixed-length windows,
`evaluation[1:] == testing[:-1]`. All 22,363 users satisfy
`training_tail == evaluation[:-1]`. Thus the last interaction is the test
target and the penultimate interaction is the validation target without
mistaking sliding-window truncation for a split error.

The source uses `sequence_data` and `user_id` from TFRecord examples. In
`NextKTokenMasking`, the final `K` tokens of the SID-token sequence become
labels and are replaced before the model forward pass. Raw final-item presence
is therefore expected; the smoke test will verify that label tokens are absent
from runtime `input_ids`.

Run on the server and save both stdout and JSON:

```bash
python tools/audit_beauty_dataset.py \
  /data2/hnlcm/dataset/amazon_data/amazon_data/beauty \
  --output logs/audit/beauty_data_audit.json | tee logs/audit/beauty_data_audit.txt
```

Acceptance checks from that JSON: contiguous item-ID domain (explicitly
zero- or one-based), no duplicate item records, zero unknown sequence item
references, sequence-length distribution consistent with the requested 20
history items plus held-out labels, no unintended duplicate user rows, and
`evaluation == testing[:-1]` for uncapped users, or
`evaluation[1:] == testing[:-1]` for the fixed-length rolling window, plus
`training_tail == evaluation[:-1]` for every user. These checks directly
validate the validation/test holdout rule without mistaking window truncation
for a split error.
