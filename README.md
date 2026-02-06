# MSS Analysis with Fine-tuned LLM

This repository contains a fine-tuned large language model (LLM) for analyzing
MSS (Membrane-type Surface Stress Sensor) data and generating odor descriptions.

---

## Model

- Gemma 3 (4B)
- Fine-tuned using LoRA
- Trained for MSS sensor data analysis

---

## Code

- `gene_open.py`  
  Generates odor descriptions from MSS signal data.

---

## Data

- `data/Sample_data.txt`  
  Contains information about:
  - Molecule concentration  
  - Receptor material  
  - MSS signal data  

  This file consists of synthetic data for demonstration purposes.

---

## Requirements

The following packages are required:

- torch  
- unsloth  
- peft  
- transformers  
- datasets  

Install them with:

```bash
pip install torch unsloth peft transformers datasets
```
## How to Run
1. Prepare input data
Prepare a text file that contains molecule concentration, receptor material, and MSS measurement data.

2. Run inference

Run the following command to perform odor prediction:

```bash
python sample_code/gene_open.py --adapter_path ./adapter_weight \
                    --data_path ./data/sample_data.txt

```
The --adapter_path argument specifies the directory containing the fine-tuned LoRA adapter.





