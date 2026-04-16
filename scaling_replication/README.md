# Scaling & Replication Evaluation

## Models (all on HuggingFace under prompterminal/)
- GPT-2 Medium: `fdm_40ch_turbo_v3_model_final`
- Qwen3-0.6B: `fdm-40ch-turbo-v3-qwen3`
- LFM2.5-1.2B: `fdm-40ch-turbo-v3-lfm2`
- Hermes3-3B (512-tok): `fdm-40ch-hermes3-3b`
- Hermes3-3B (1024-tok): `fdm-40ch-hermes3-1024tok`

## Test Data Generation
```bash
# GPT-2
python3 eidetic_real_fdm_e2e_40ch_turbo_v3.py generate

# Qwen3
python3 eidetic_qwen3_fdm_e2e.py generate

# LFM2.5
python3 eidetic_lfm2_fdm_e2e.py generate

# Hermes3 512-tok
python3 eidetic_hermes3_fdm_e2e.py generate

# Hermes3 1024-tok
python3 eidetic_hermes3_1024tok.py generate
```

## Evaluation Commands
```bash
python3 eval_gpt2.py 2>&1 | tee results/gpt2_eval_rerun.log
python3 eval_qwen3.py 2>&1 | tee results/qwen3_eval_rerun.log
python3 eval_lfm2.py 2>&1 | tee results/lfm2_eval_rerun.log
python3 eval_hermes3.py 2>&1 | tee results/hermes3_eval_rerun.log
python3 eidetic_hermes3_1024tok.py eval 2>&1 | tee results/hermes3_1024_full_eval.log
```

## Results (Table 1)
| Model | Action | Rule* | Meta | Fact | Extra |
|-------|--------|-------|------|------|-------|
| GPT-2 Medium (115K) | 98.7% | 39.1% | 100.0% | 100.0% | 99.7% |
| Qwen3-0.6B | 98.7% | 100.0% | 100.0% | 100.0% | 100.0% |
| LFM2.5-1.2B | 98.6% | 100.0% | 100.0% | 100.0% | 93.0% |
| Hermes3-3B (512-tok) | 98.9% | 100.0% | 100.0% | 100.0% | 100.0% |
| Hermes3-3B (1024-tok) | 99.0% | 100.0% | 100.0% | 100.0% | 99.8% |

*Rule reported for META=NONE samples only.
