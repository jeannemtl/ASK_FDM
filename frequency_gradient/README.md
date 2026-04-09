# Frequency Gradient Evaluation (Table 6)

Per-channel retrieval accuracy across carrier frequency bands (ch8--39).

## GPT-2, Qwen3, LFM2.5

Per-channel accuracy for these three architectures is computed as a
byproduct of the joint retrieval eval script. Run:

```bash
python joint_accuracy_multi.py gpt2
python joint_accuracy_multi.py qwen3
python joint_accuracy_multi.py lfm2
```

The script prints per-channel accuracy for all 32 extra channels (ch8--39)
grouped by carrier frequency band. Update `MODEL_CONFIGS` at the top of
the script to point to your local model checkpoints and test data.

## Hermes3

Hermes3 per-channel accuracy uses a dedicated script:

```bash
python hermes3_perchannel_eval.py
```

See `hermes3_perchannel_eval.py` for model path configuration.
