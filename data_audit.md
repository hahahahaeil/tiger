# Beauty data audit

Status: filesystem audit completed locally; record-level audit is pending the
server `grid` environment because this dataset is GZIP TFRecord.

The actual local Beauty directory is
`D:/data/amazon_data/amazon_data/beauty`, one level below the supplied root.
The intended server counterpart is
`/data2/hnlcm/dataset/amazon_data/beauty`.

| Split | Files | Bytes |
| --- | ---: | ---: |
| items | 24 | 35,963,432 |
| training | 175 | 417,282,085 |
| evaluation | 175 | 435,615,164 |
| testing | 175 | 496,831,085 |
| total | 550 | 1,385,699,962 |

The SHA-256 of the deterministic local `size + relative-path` manifest is
`41F90D273E347D90946686B4A6B037965289D091B0B69856AF2B151A2C88F241`.
This is a layout fingerprint, not a content hash. The server run must record
its own manifest and compare file counts and total bytes.

The four required directories exist locally: `items`, `training`,
`evaluation`, and `testing`. The source uses `sequence_data` and `user_id`
from TFRecord examples. In `NextKTokenMasking`, the final `K` tokens of the
SID-token sequence become labels and are replaced before the model forward
pass. Therefore raw final-item presence is expected; the runtime audit must
verify that those label tokens are absent from `input_ids`.

Run on the server and save both stdout and JSON:

```bash
python tools/audit_beauty_dataset.py \
  /data2/hnlcm/dataset/amazon_data/beauty \
  --output logs/audit/beauty_data_audit.json | tee logs/audit/beauty_data_audit.txt
```

Acceptance checks from that JSON: contiguous item-ID domain (explicitly
zero- or one-based), no duplicate item records, zero unknown sequence item
references, sequence-length distribution consistent with the requested 20
history items plus held-out labels, and no unintended duplicate user rows.
