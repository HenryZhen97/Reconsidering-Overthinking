<div align="center">

# rLLM

<div>
🚀 Democratizing Reinforcement Learning (RL) for LLMs 🌟
</div>
</div>
<div>
<br>

<div align="center">

[![Github](https://img.shields.io/badge/RLLM-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/agentica-project/rllm)
[![Website](https://img.shields.io/badge/Site-%23000000.svg?style=for-the-badge&logo=semanticweb&logoColor=white)](https://www.agentica-project.com) 
[![Twitter](https://img.shields.io/badge/Agentica-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/Agentica_)
[![Hugging Face Collection](https://img.shields.io/badge/Agentica-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/agentica-org)

</div>

</div>


## Overview

rLLM is an open-source project to fully democratize reinforcement learning (RL) for LLMs and reproduce DeepSeek R1 and OpenAI O1/O3 at scale on real tasks. For all releases, we open source all our efforts here-including training scripts (including hyperparameters), models, systems, dataset, and logs. 

<div align="center">
<img src="figures/deepcoder.png" width="60%" />

<sub>*DeepCoder's LiveCodeBench (LCB) score as training progresses. At step 180, context length is extended to 32K. The best 32K checkpoint is used for inference-time scaling to 64K, achieving 60.6% LCB—matching o3-mini's performance. For more details, see our [blog post](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51).*</sub>
</div>

## Releases  📰

<strong>[2025/04/08]</strong> We release `DeepCoder-14B-Preview`, a 14B coding model that achieves an impressive **60.6%** Pass@1 accuracy on LiveCodeBench (+8% improvement), matching the performance of `o3-mini-2025-01-031 (Low)` and `o1-2024-12-17`. As part of this release, we open-source:
- ⬆️ An In-Depth Blog Post on our [Training Recipe and Insights](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51)
- 🤗 HF Model [`DeepCoder-14B-Preview`](https://huggingface.co/agentica-org/DeepCoder-14B-Preview), [`DeepCoder-1.5B-Preview`](https://huggingface.co/agentica-org/DeepCoder-1.5B-Preview)
- 🤗 HF Dataset [`DeepCoder-Preview-Dataset`](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset)
- 📄 [Training Scripts](https://github.com/agentica-project/rllm/tree/main/scripts/deepcoder/train)—Exact hyperparameters we used to achieve `o3-mini` performance.
- 📈 [Wandb Training Logs](https://wandb.ai/mluo/deepcoder)—All training runs and ablations.
- 🔎 [Evaluation Logs](https://drive.google.com/file/d/1tr_xXvCJnjU0tLO7DNtFL85GIr3aGYln/view?usp=sharing)—LiveCodeBench and Codeforces logs for DeepCoder.

<strong>[2025/02/10]</strong> We release `DeepScaleR-1.5B-Preview`, a 1.5B model that surpasses O1-Preview and achieves <strong>43.1% Pass@1</strong> on AIME. We achieve this by iteratively scaling Deepseek's GRPO algorithm from 8K→16K->24K context length for thinking. As part of this release, we open-source:

- 🍗 An In-Depth Blog Post on our [Training Recipe and Insights](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)
- 🤗 HF Model [`DeepScaleR-1.5B-Preview`](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview)
- 🤗 HF Dataset [`DeepScaleR-Preview-Dataset`](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) / 🗂️  [JSON Dataset](https://github.com/agentica-project/deepscaler/tree/main/deepscaler/data)
- 📄 [Training Scripts](https://github.com/agentica-project/deepscaler/tree/main/scripts/train)—Exact hyperparameters we used to achieve 43.1% on AIME.
- 📈 [Wandb Training Logs](https://wandb.ai/mluo/deepscaler-1.5b)—All training runs and ablations.
  - Due to Wandb migration bugs, the 8k training run is compressed to 400-500 steps. The data is identical, but our original run was 1600 steps.
- 🔎 [Evaluation Logs](https://drive.google.com/file/d/1V_rYKoL35WmubbmWN6PeFg4zo5QOug8X/view?pli=1)—DeepScaleR, Deepseek Distill, and Still 1.5B generations over 1000+ math problems.


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

