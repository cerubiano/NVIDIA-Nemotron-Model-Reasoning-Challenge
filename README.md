# NVIDIA Nemotron Model Reasoning Challenge

Kaggle competition solution for the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge).

## Approach

Fine-tuning the **Nemotron-3-Nano-30B-A3B** model with LoRA (rank 16) using category-specific chain-of-thought (CoT) training data generated for 6 puzzle types.

### Key insight
Three of the six puzzle categories (physics equations, unit conversion, Roman numerals) have **100% deterministic solutions** — the model learns to infer hidden parameters from examples and apply exact formulas. This guarantees near-perfect accuracy on 50% of the test set.

### Dataset
8,132 training examples with chain-of-thought reasoning across 6 categories:

| Category | Examples | Generation method |
|---|---|---|
| Physics equation | 1,597 | Python (analytical) |
| Unit conversion | 1,594 | Python (analytical) |
| Numeral system | 1,576 | Python (analytical) |
| Bit manipulation | 1,129 | Claude Haiku API |
| Text cipher | 1,122 | Claude Haiku API |
| Algebraic symbols | 1,114 | Claude Haiku API |

Training data available on Kaggle: [nemotron-training-data](https://www.kaggle.com/datasets/carlosrubianorobles/nemotron-training-data)

## Pipeline

```
scripts/
├── 1_install.sh          # Install dependencies (RunPod A100)
├── 2_download_data.py    # Download training data from Kaggle
├── 3_train.py            # SFT training with LoRA r=16
└── 4_package.py          # Package adapter into submission.zip
```

## Setup & Training

### Requirements
- GPU: A100 80GB (or equivalent)
- CUDA 12.x
- Python 3.11+

### Run

```bash
# 1. Clone repo
git clone https://github.com/cerubiano/NVIDIA-Nemotron-Model-Reasoning-Challenge
cd NVIDIA-Nemotron-Model-Reasoning-Challenge

# 2. Install dependencies (~5 min)
bash scripts/1_install.sh

# 3. Download training data
python3 scripts/2_download_data.py

# 4. Train (~4-6 hours on A100 80GB)
python3 scripts/3_train.py

# 5. Package submission
python3 scripts/4_package.py
```

## Model

- Base: `unsloth/Nemotron-3-Nano-30B-A3B`
- LoRA rank: 16 (max allowed: 32)
- Training: SFT with chain-of-thought data
- Epochs: 2
- Learning rate: 1e-4

## Results

| Submission | Score | Notes |
|---|---|---|
| v1 | TBD | Baseline SFT |

## License

CC BY 4.0 — as required by the competition rules.
