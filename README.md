# Building an LLM from Scratch

> A step-by-step implementation of a GPT-like large language model in PyTorch, built as a hands-on learning project following the book *Build a Large Language Model (From Scratch)* by Sebastian Raschka.

## What This Project Covers

- Tokenization (BPE) from scratch
- Multi-head attention mechanism
- GPT model architecture
- Pretraining on unlabeled text
- Finetuning for text classification
- Instruction finetuning (chat-like behavior)

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Project Structure

```
├── notebooks/         # Jupyter notebooks for each chapter
├── src/               # Clean, importable Python modules
│   ├── config.py      # Model hyperparameters
│   ├── tokenizer.py   # BPE tokenizer
│   ├── attention.py   # Multi-head attention
│   ├── model.py       # GPT model class
│   ├── train.py       # Pretraining loop
│   ├── finetune.py    # Classification + instruction tuning
│   └── generate.py    # Text generation
├── data/              # Datasets (not tracked in git)
├── checkpoints/       # Model weights (not tracked in git)
└── tests/             # Unit tests
```

## Hardware

Built and trained on an NVIDIA RTX 4050 (6GB VRAM) with PyTorch CUDA.

## Acknowledgments

Based on the book *Build a Large Language Model (From Scratch)* by Sebastian Raschka.  
Reference repository: [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
