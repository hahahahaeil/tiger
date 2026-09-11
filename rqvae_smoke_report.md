# Beauty RQ-VAE smoke report

Status: passed on 2026-09-11. This is a GRID-setting smoke test, not a formal
TIGER result: it uses one embedding shard (1,024 items), three optimization
steps, three RQ layers, and codebook width 16.

## Inputs and output

- embedding input:
  `logs/audit/beauty_smoke_embedding_local_model_retry2/pickle/merged_predictions_tensor.pt`
- embedding shape: `[1024, 2048]`, validated finite before training
- checkpoint:
  `logs/audit/beauty_smoke_rqvae_retry1/checkpoints/checkpoint_000_000003.ckpt`
- seed: 42

## Evidence

The trainer reported `max_steps=3` reached and the restart callback reported
that the job finished successfully. The checkpoint callback saved at global step
3. Final reported metrics were all finite: `train/loss_epoch=0.4206`,
`train/quantization_loss_epoch=0.4202`,
`train/reconstruction_loss_epoch=0.000381`, and
`train/mse_epoch=0.000417`.

The early zero loss entries occurred during layer/codebook initialization. The
three optimization steps subsequently reported finite loss values, including
the final `train/loss_step=0.000641`,
`train/quantization_loss_step=1.8087e-05`, and
`train/reconstruction_loss_step=0.000623`. The log also reports RQ codebook
coverage and entropy for each of three layers. No RQ-KMeans component was run.

## Next gate

Load the saved RQ-VAE checkpoint through `beauty_smoke_rqvae_inference` and
export the raw `N x 3` RQ assignments, then the existing deduplication and
transpose post-processors must produce the TIGER-facing `4 x N` SID map. The
SID audit must confirm the codebook range, fourth collision digit, uniqueness,
and reversibility before TIGER forward/beam smoke is allowed.
