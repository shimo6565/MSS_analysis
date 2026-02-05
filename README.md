# MSS_analysis (Graduation Research)

LoRA fine-tuned Gemma-3 (4B) adapter for MSS sensor analysis.

## Tasks
- MSS_TO_MOLECULE
- MOLECULE_TO_ODOR
- MSS_TO_ODOR (combined)

## Included artifacts
- `adapter_model.safetensors` (Git LFS)
- `adapter_config.json`

## Demo (example)
```bash
python sample_code/inference.py --adapter ./ --task MSS_TO_ODOR --mss "..."


