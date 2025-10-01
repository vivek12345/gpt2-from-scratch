# GPT-2 Implementation from Scratch

A clean, educational implementation of GPT-2 built from scratch using PyTorch. This project demonstrates the architecture and training of transformer-based language models.

## 🎯 Features

- **Complete GPT-2 Architecture**: Full implementation including multi-head attention, feed-forward layers, and layer normalization
- **Multiple Attention Mechanisms**: Self-attention, causal attention, and multi-head attention
- **Training Pipeline**: Complete training loop with validation and loss tracking
- **Text Generation**: Sampling with temperature and top-k filtering
- **Pretrained Weights**: Download and use official OpenAI GPT-2 weights
- **Educational Examples**: Clear demonstrations of attention mechanisms and model usage

## 📁 Project Structure

```
llm-from-scratch-gpt2/
├── src/                          # Main source code
│   ├── models/                   # Model architectures
│   │   ├── gpt2.py              # Main GPT-2 model
│   │   └── block.py             # Transformer block
│   ├── layers/                   # Neural network layers
│   │   ├── multihead_attention.py
│   │   ├── causal_attention.py
│   │   ├── self_attention.py
│   │   ├── feed_forward.py
│   │   ├── layer_norm.py
│   │   └── rms_norm.py
│   ├── data/                     # Data handling
│   │   ├── data_loader.py
│   │   └── tokenization.py
│   ├── training/                 # Training utilities
│   │   ├── train.py
│   │   └── loss.py
│   └── utils/                    # Utility functions
│       ├── text_generation.py
│       └── model_utils.py
├── scripts/                      # Executable scripts
│   ├── train_gpt2.py            # Training script
│   └── download_gpt2_weights.py # Download pretrained weights
├── examples/                     # Usage examples
│   ├── basic_usage.py
│   └── attention_demo.py
├── data/                         # Training data
│   └── verdict.txt
└── gpt2/                         # Downloaded GPT-2 weights (gitignored)
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd llm-from-scratch-gpt2
```

2. Install dependencies (using uv or pip):
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Basic Usage

```python
import tiktoken
import torch
from src.models.gpt2 import GPTModel
from src.utils.text_generation import generate_text_simple
from src.data.tokenization import token_ids_to_text

# Initialize model
GPT2_CONFIG = {
    "vocab_size": 50257,
    "emb_dim": 768,
    "context_length": 1024,
    "num_layers": 12,
    "num_heads": 12,
    "qkv_bias": True,
    "dropout": 0.1
}

model = GPTModel(GPT2_CONFIG)
tokenizer = tiktoken.get_encoding("gpt2")

# Generate text
output = generate_text_simple(
    model=model,
    tokenizer=tokenizer,
    start_context="Once upon a time",
    context_length=GPT2_CONFIG["context_length"],
    max_new_tokens=50,
    temperature=1.0
)

print(token_ids_to_text(output, tokenizer))
```

### Training

Train your own GPT-2 model:

```bash
uv run scripts/train_gpt2.py
```

### Download Pretrained Weights

Download official OpenAI GPT-2 weights:

```bash
uv run scripts/download_gpt2_weights.py
```

## 📚 Examples

### Basic Usage
```bash
uv run examples/basic_usage.py
```

Shows:
- Forward pass through the model
- Text generation with sampling
- Model statistics

### Attention Mechanisms
```bash
uv run examples/attention_demo.py
```

Demonstrates:
- Self-attention (no masking)
- Causal attention (masked)
- Multi-head attention

## 🏗️ Architecture

### GPT-2 Model Components

1. **Token & Positional Embeddings**: Convert tokens to dense vectors with position information
2. **Transformer Blocks** (x12):
   - Multi-head self-attention with causal masking
   - Layer normalization
   - Feed-forward network (expand to 4x, then project back)
   - Residual connections
3. **Final Layer Norm**: Stabilize outputs
4. **Output Projection**: Map to vocabulary logits

### Configuration

The model uses the standard GPT-2 (124M) configuration:
- **Vocabulary Size**: 50,257 tokens
- **Embedding Dimension**: 768
- **Context Length**: 1024 tokens
- **Transformer Layers**: 12
- **Attention Heads**: 12 (64 dimensions each)
- **Feed-Forward Hidden Size**: 3072 (4 × 768)

## 🔬 Technical Details

### Attention Mechanisms

The project includes three attention implementations:

1. **Self-Attention**: Basic attention without masking (educational)
2. **Causal Attention**: Masked attention for autoregressive generation
3. **Multi-Head Attention**: Parallel attention heads for richer representations

### Training Features

- **Cross-Entropy Loss**: Standard language modeling objective
- **AdamW Optimizer**: Weight decay for regularization
- **Learning Rate**: 3e-4 (adjustable)
- **Gradient Accumulation**: Configurable batch sizes
- **Validation**: Track train/val loss during training

### Text Generation

- **Temperature Sampling**: Control randomness (0 = greedy, >1 = more random)
- **Top-K Sampling**: Sample from top K most likely tokens
- **Early Stopping**: Optional end-of-sequence token

## 📊 Model Statistics

- **Total Parameters**: ~124M (117M for 124M config)
- **Model Size**: ~475 MB (float32)
- **Context Window**: 1024 tokens
- **Training Speed**: ~X tokens/second (depends on hardware)

## 🛠️ Development

### Running Tests

```bash
# Test basic usage
uv run examples/basic_usage.py

# Test attention mechanisms
uv run examples/attention_demo.py

# Test training loop
uv run scripts/train_gpt2.py
```

### Code Style

The project follows:
- Clear, educational code with comments
- Type hints where helpful
- Modular architecture
- Docstrings for all functions

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Inspired by Sebastian Raschka's "Build a Large Language Model from Scratch"
- OpenAI for the original GPT-2 architecture and weights
- The PyTorch team for the excellent framework

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

## 🔗 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/) - Visual guide to GPT-2
