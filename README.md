# Verne Decoder Transformer

This repository contains an implementation of a Transformer Decoder in PyTorch from scratch. The purpose of this project is to generate text based on Jules Verne's literary works, using the original Transformer model as proposed in the ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) paper and its subsequent improvements.

&nbsp;
## 📋 Plan

Our plan includes several key steps:

- ✅ Start with a basic bigram model and a basic table lookup embedding layer. After **10,000 iterations**, the results are as follows:

   | Metric     |  Value |
   | :--------- | -----: |
   | Train Loss | 2.4980 |
   | Val Loss   | 2.5421 |

- 🔄 Add a self-attention block.

- 🔲 Implement self-attention heads and introduce basic positional embeddings.

- 🔲 Add a feed-forward network and stack multiple blocks of multi-head attention.

- 🔲 Integrate residual connections.

- 🔲 Implement Layer Normalization.

- 🔲 Experiment with different tokenizers, including TikToken, HuggingFace's Tokenizers library, and BPE-based tokenizers.

- 🔲 Migrate to a proper data loader for more efficient data handling. Compare DataLoader and IterableDataset from PyTorch and decide which is more suitable for our case.

- 🔲 Implement the original positional encoding scheme as proposed in the original Transformer paper. Experiment with absolute and relative positional encodings.

- 🔲 Compare our implementation with the official PyTorch Transformer Decoder implementation. Evaluate the performance, computational efficiency, and flexibility of both models.

- 🔲 Implement popular improvements to the original Transformer model:

  - Implement Sparse Attention mechanism as described in ["Transformers are Sparse"](https://arxiv.org/abs/2104.04473) to make the model more efficient and capable of handling longer sequences.
  
  - Experiment with the recently introduced SWiLu (Sine Windowed Linear Unit) activation function, as it's been shown to improve Transformer's performance in some cases. (Please note that at the time of this response (September 2021), there's no formal paper presenting SWiLu, thus no link is provided.)
  
  - Look into other modifications of the Transformer architecture that aim to improve the handling of long sequences, computational efficiency, or modeling capacity.


&nbsp;
## ▶ Usage

To use this code, first, clone the repository:

```bash
git clone https://github.com/joaoflf/transformer_decoder_pytorch.git
cd transformer_decoder_pytorch
```

Next, install the dependencies:

```bash
pip install -r requirements.txt
```

You can then run the main script:

```bash
python main.py
```