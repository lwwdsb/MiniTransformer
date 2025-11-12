Mini-Transformer: A Minimal PyTorch Implementation of Transformer for NLP Tasks
Overview

This project is a from-scratch implementation of a mini version of the Transformer model using PyTorch, designed to help understand the core mechanisms behind modern sequence-to-sequence models such as BERT, GPT, and T5.
It includes essential components like multi-head attention, positional encoding, encoder-decoder architecture, and a simple training & inference pipeline.

This project was built as a hands-on exploration of deep learning for NLP — lightweight, educational, and fully readable.

Project Structure
MiniTransformer/
│
├── train.py           # Training loop and loss optimization
├── model.py           # Transformer architecture (encoder, decoder, attention)
├── vocab.py           # Vocabulary builder and tokenizer utilities
├── toy_data.py        # Toy parallel dataset for testing translation
├── inference.py       # Greedy decoding for inference
└── README.md          # Documentation (this file)

Installation
Clone the repository
git clone https://github.com/<your-username>/MiniTransformer.git
cd MiniTransformer

Create a conda or virtual environment
conda create -n transformer python=3.10
conda activate transformer

Install dependencies
pip install torch tqdm numpy

Usage
Training

Train the Mini-Transformer on the toy dataset:

python train.py


You’ll see logs showing loss decreasing over epochs.

Inference

Run greedy decoding to generate translations:

python
>>> from inference import greedy_decode
>>> from model import MiniTransformer
>>> from vocab import src_vocab, tgt_vocab, tgt_idx2word
>>> import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = MiniTransformer(len(src_vocab), len(tgt_vocab)).to(device)
model.load_state_dict(torch.load("checkpoint.pth"))
print(greedy_decode(model, "one two three", src_vocab, tgt_vocab, tgt_idx2word, device))

Model Architecture

Embedding layer: Converts tokens into dense vectors.

Positional Encoding: Adds sequence order information.

Multi-Head Self-Attention: Enables contextual understanding.

Feed-Forward Network: Applies non-linear transformation.

Encoder-Decoder stack: Classic Transformer structure.

Greedy Search: Simple decoding strategy for translation generation.

All components are implemented from scratch — no reliance on nn.Transformer from PyTorch.

Example Output
Input:   one two three
Output:  uno dos tres


(Toy example — model trained on synthetic English-Spanish pairs.)