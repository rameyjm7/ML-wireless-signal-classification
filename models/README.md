# Model Artifacts

This public repository does not include trained model weights, TensorRT engines,
ONNX exports, checkpoints, generated IQ samples, or calibration captures.

The public repo provides source code, training/evaluation workflows, export
scripts, model cards, and result summaries. Bring your own trained artifacts,
train models with the provided workflows, or contact Jacob Ramey to license
validated model artifacts and commercial integrations.

## Layout

```text
models/
  README.md
  noisy_drone_rf_v2/
    labels.json              Public class labels only.
    run_onnx_inference.sh    Example runner; requires a local/private model.
```

## Expected Local/Private Artifacts

Common artifact types intentionally excluded from public Git:

- `*.keras`, `*.h5`, `*.onnx`, `*.engine`
- model checkpoints and TensorRT build outputs
- generated or captured IQ: `*.npy`, `*.npz`, `*.bin`, `*.c64`, `*.iq`
- replay manifests that point to private captures
- production calibration data and deployment configs

## Usage Pattern

```bash
# Train or copy a private/licensed model artifact locally.
python exports/export_noisy_drone_to_onnx.py \
  --keras-model /path/to/private/model.keras \
  --out /path/to/private/model.onnx
```

```bash
# Run inference with a local/private artifact.
python exports/run_onnx_inference.py \
  --onnx /path/to/private/model.onnx \
  --iq-file /path/to/local/sample.npy \
  --labels models/noisy_drone_rf_v2/labels.json
```

Commercial licensing, validated model artifacts, and integration support:

- Jacob Ramey: rameyjm7@gmail.com
