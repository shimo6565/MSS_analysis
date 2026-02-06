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

- `data/sample_data.txt`  
  Contains information about:
  - Molecule concentration  
  - Receptor material  
  - MSS signal data  

  This file consists of fictitious (synthetic) data for demonstration purposes.

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
How to Run
1. Prepare input data
Prepare a text file that contains MSS measurement data.

Example:

data/sample_data.txt
2. Run inference
Execute the following command:

python gene_open.py --data_path ./data/sample_data.txt
The data_path argument specifies the path to a text file that contains:

Molecule concentration

Receptor material

Signal data

Notes
The included sample data is synthetic and does not contain real experimental data.

This repository is intended for research and demonstration purposes.

Author
Takeshi Shimoshige




