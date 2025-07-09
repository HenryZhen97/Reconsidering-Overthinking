<div align="center">
  
# Recosidering Overthinking: Penalizing Internal and External Redundancy in CoT Reasoning

</div>

## Overview

we revisit overthinking by decomposing it into two distinct forms of redundancy: **internal redundancy**, which originates within FCS, and **external redundancy**, which emerges after the FCS has been reached. To address both forms of redundancy, we propose a dual-penalty mechanism tailored to their characteristics in reinforcement learning framework. For internal redundancy, we introduce a sliding-window semantic similarity strategy that detects and penalizes semantically repetitive spans, encouraging more informative and concise reasoning. For external redundancy, we quantify its severity by measuring its proportional length relative to the entire reasoning trace and apply an equivalent penalty. 


## Getting Started 🎯
### Installation
```bash
# Installing Python 3.10 Environment.
conda create -n rllm python=3.10 -y
conda activate rllm

# Installing RLLM dependencies.
cd rllm
pip install -e ./verl
pip install -e .
```

### Data
Our raw training data is in `rllm/data/[train|test]/[code|math]/`, along with preprocessing scripts in `rllm/data/preprocess`. To convert the raw data into Parquet files for training, run:

```bash
# Download datasets from GDrive, populates rllm/data/[train|test]/[math|code]/*.json
python scripts/data/download_datasets.py

# Generate parquet files
python scripts/data/deepscaler_dataset.py
```

### Training Scripts

We provide training scripts in the `scripts/deepscaler/train/`. To fully reproduce our CoT Compress results, please refer to the corresponding `README.md` files in each directory.

## Evaluation ⚖️

Our evaluation scripts automatically runs many replicas of vLLM. To run our evaluation scripts, run:
```bash
./scripts/eval/eval_model.sh
```

## Citation

