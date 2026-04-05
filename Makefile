serve_pi05_droid:
	XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
  uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid_jointpos_polaris \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_droid_jointpos

serve_pi0:
	XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
  uv run python scripts/serve_policy.py --port 8000 \
  policy:checkpoint \
  --policy.config=pi0_fast_droid_jointpos \
  --policy.dir=s3://openpi-assets-simeval/pi0_fast_droid_jointpos
