# TIGER Beauty smoke-test report

Status: ready to run. The server-side dataset audit has passed and no model
code has been changed. The two dedicated smoke configurations constrain the
embedding and RQ-VAE stages to one item shard, one GPU, and three RQ-VAE
updates.

The smoke sequence is embedding -> RQ-VAE -> SID export/deduplication -> TIGER
forward/training -> constrained beam generation -> Recall/NDCG. It must use a
separate output directory and at most a few files and steps.

Required recorded checks:

1. embedding tensor has one row per item and finite values;
2. RQ-VAE loss and reconstruction MSE are finite and at least one gradient
   norm is non-zero;
3. raw RQ-VAE IDs have shape `N x K`, whereas the TIGER loader receives `K x N`;
4. the fourth deduplication digit exists, every SID is in vocabulary range, and
   `item -> SID -> item` is one-to-one;
5. TIGER forward loss is finite and gradients are non-zero;
6. constrained beam search returns valid SIDs and reports invalid-ID ratio;
7. evaluator emits Recall@5, NDCG@5, Recall@10, and NDCG@10.

Configuration repair applied, pending smoke validation: this repository
contained `rqvae_train_flat.yaml` but no RQ-VAE inference/export configuration.
The README instead documents RQ-KMeans inference. `rqvae_inference_flat.yaml`
now composes the existing RQ-VAE training architecture and changes only the
inference datamodule/callbacks. It reuses `ResidualQuantization.predict_step`,
`LocalPickleWriter`, and the existing post-processors (deduplicate then
transpose). Model architecture is unchanged. The export result must still be
validated for `N x K` before transpose and `K x N` after transpose.

The embedding smoke deliberately uses the repository's configured
`google/flan-t5-xl` encoder. Its embedding size must be 2,048; using a smaller
FLAN variant would invalidate the subsequent GRID RQ-VAE shape check. This is
still a GRID setting rather than a strict TIGER reproduction, because the
TIGER paper uses Sentence-T5.
