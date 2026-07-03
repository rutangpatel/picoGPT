# picoGPT

![License](https://img.shields.io/badge/license-MIT-green) 
![Version](https://img.shields.io/badge/version-1.0.0-blue) 
![Language](https://img.shields.io/badge/language-Python-yellow) 
![Framework](https://img.shields.io/badge/framework-TensorFlow-orange) 
![GitHub](https://img.shields.io/badge/GitHub-rutangpatel/picoGPT-black?logo=github)
![Kaggle](https://img.shields.io/badge/Kaggle-rutangpatel-blue?logo=kaggle)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Training](#training)
- [Results](#results)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)

## Project Information

- **Author:** rutangpatel
- **Kaggle:** [rutangpatel](https://www.kaggle.com/rutangpatel)
- **Version:** 1.0.0
- **License:** MIT
- **Repository:** [https://github.com/rutangpatel/picoGPT](https://github.com/rutangpatel/picoGPT)
- **Keywords:** gpt transformer decoder-only tensorflow keras character-level tinystories language-model from-scratch causal-attention next-token-prediction

## Features

- Decoder-only Transformer (GPT-style) built from scratch in TensorFlow/Keras
- Causal (masked) self-attention with no peeking at future tokens
- Stacked decoder blocks with multi-head attention, residual connections, and layer normalization
- Character-level tokenization with save/load support
- Autoregressive text generation with temperature sampling
- Trained on TinyStories dataset for coherent short-story generation
- Checkpointing every epoch
- Mixed precision training support (FP16)
- Multi-GPU ready with tf.distribute.MirroredStrategy

## Installation

1. Clone the repository
2. Create a virtual environment and install dependencies from requirements.txt
3. Download the TinyStories dataset (auto-downloaded on first run via HuggingFace datasets)
4. Run the tokenizer to build vocab.json and encoded_train_data.npy
5. Run train.py to train the model
6. Run generate.py to generate stories

```bash
git clone https://github.com/rutangpatel/picoGPT.git
cd picoGPT
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tokenizer.py
python train.py
python generate.py
```

## Usage

### Build Tokenizer and Encode Data

```bash
python tokenizer.py
```

This loads the TinyStories dataset, builds a character-level vocabulary, saves vocab.json, and encodes the full dataset into encoded_train_data.npy (~3.5GB).

### Train the Model

```bash
python train.py
```

Default hyperparameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| SEQ_LEN | 128 | Context window length |
| BATCH_SIZE | 32 | Training batch size |
| EPOCHS | 5 | Number of training epochs |
| D_MODEL | 256 | Embedding dimension |
| NUM_HEADS | 8 | Attention heads |
| NUM_MODELS | 6 | Decoder blocks (layers) |
| DFF | 1024 | Feed-forward hidden dimension |
| DROPOUT | 0.1 | Dropout rate |

### Generate Stories

```bash
python generate.py
```

Example output:

```
Prompt: Once upon a time
----------------------------------------
Once upon a time, there were many trees and flowers and animals.

"Wow, look at that!" Lily said, pointing to the window.

"That is the surprise, mom?" Tom asked.

"Yes, it is. This hotel is famous because it is next to the park. You can see all the animals from here. Do you want to go and see them?" mom asked.

"Yes, yes, yes!" Tom and Lily said.

They put on their coats and shoes and went downstairs. They walked to the park and saw many animals. They saw lions and tigers and bears and monkeys and elephants an ....
```

## Architecture

```
Input Tokens
    |
    |---> Token Embedding (vocab_size -> d_model)
    |---> Positional Embedding (max_seq_len -> d_model)
    |         ---> Add together
    |
    |---> [Decoder Block] x 6
            |
            |---> Multi-Head Causal Self-Attention
            |         query = key = value = input
            |         use_causal_mask = True
            |         ---> Residual + LayerNorm
            |
            |---> Feed-Forward Network (d_model -> dff -> d_model)
            |         ---> Residual + LayerNorm
            |
    |---> Dropout
    |
    |---> Dense Layer (d_model -> vocab_size)
            ---> Softmax over vocabulary
```

### Decoder Block Details

Each decoder block contains:

1. Multi-Head Self-Attention - attends to all previous positions with causal masking
2. Residual Connection + Layer Normalization - preserves gradient flow
3. Feed-Forward Network - two linear layers with ReLU: d_model -> dff -> d_model
4. Dropout - regularization during training

### Key Design Decisions

| Choice | Reason |
|--------|--------|
| Character-level tokenization | Simplicity - no external tokenizer library needed |
| Learned positional embeddings | Easier to implement than sinusoidal encoding |
| Causal masking in attention | Ensures autoregressive generation (no peeking at future tokens) |
| Post-LayerNorm | Original Transformer design; stable for small models |

## Training

### Dataset

- **Source:** [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- **Size:** ~2.1M short stories
- **Content:** Simple English stories, 2-5 sentences each, written for 3-4 year olds
- **Encoded tokens:** ~1.9B characters

### Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | Adam (lr = 3e-4) |
| Loss | Sparse Categorical Crossentropy (from_logits=True) |
| Metric | Sparse Categorical Accuracy |
| Hardware | NVIDIA P100 (Kaggle) |
| Time per epoch (10M chars, batch=32, seq=128) | ~3.4 hours/epoch |

### Observations

- Loss decreases steadily from ~0.35 to ~0.25 over 2 epochs
- Training accuracy reaches 92.48% on next-character prediction
- Model learns: grammar, dialogue formatting, story structure, character names
- Limitation: With SEQ_LEN=128, stories drift after ~128 characters (context window limit). Additionally, training on only 10M characters leads to repetitive names like Tom and Lily appearing across unrelated stories.

## Results

### What picoGPT Learned

| Capability | Example |
|-----------|---------|
| Grammar | Proper sentence structure, punctuation |
| Dialogue | "Mom, can I have the robot, please?" |
| Story structure | Setup -> conflict -> resolution |
| Named entities | Characters like "Lily", "Tom", "Max" |
| Cause-effect | "The pig was too heavy... Can you help me?" |

### Known Limitations

| Issue | Cause | Fix |
|-------|-------|-----|
| Stories drift after ~128 chars | Context window too short | Increase SEQ_LEN to 256/512 |
| Repetitive names (Tom, Lily everywhere) | Small training subset (10M chars) | Train on 50M+ chars |
| Character-level spelling mistakes | Tokenizing by letter, not word | Upgrade to BPE tokenizer |
| Characters switch mid-story | Lost context at window boundary | Longer SEQ_LEN + more data |

## Project Structure

```
picoGPT/
|
|-- model.py              # GPTModel & DecoderBlock architecture
|-- tokenizer.py          # Character-level tokenizer
|-- train.py              # Training script
|-- generate.py           # Text generation script
|-- dataset.py            # TinyStories dataset loader
|
|-- vocab.json            # Saved vocabulary (generated)
|-- encoded_train_data.npy # Encoded dataset (generated, ~3.5GB)
|-- picoGPT.weights.h5    # Saved model weights (generated)
|
|-- .gitignore            # Excludes large generated files
|-- requirements.txt      # Python dependencies
|-- README.md             # This file
```

## Tech Stack

- **Deep Learning:** TensorFlow, Keras
- **Dataset:** HuggingFace datasets (TinyStories)
- **Numerical:** NumPy
- **Tokenizer:** Custom character-level (BPE planned for v2)
- **Hardware:** NVIDIA P100 (Kaggle) / NVIDIA L4 (e2e networks - planned)

## Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you would like to change.

### Future Improvements

- BPE Tokenization - 4-5x more efficient than character-level
- Increase SEQ_LEN to 512 - coherent stories up to ~2000 characters
- Train on 50M-100M characters - eliminate repetitive names and plots
- Top-k / Top-p sampling - reduce repetition in generation
- Repetition penalty - actively discourage reusing recent tokens
- Validation split - monitor overfitting, implement early stopping
- Mixed precision (FP16) - faster training, lower memory
- Multi-GPU training - tf.distribute.MirroredStrategy for 2x+ speedup
- Gradient accumulation - simulate larger batches without more VRAM

---

Built with curiosity, patience, and way too many hours debugging tf.concat shapes.

- rutangpatel
- Trained on Kaggle P100. Future versions planned on e2e networks L4.