# 🧠 Memorization Capabilities of Neural Networks: Understanding Scaling Laws in Autoregressive and Sequence-To-Sequence Learning

**Investigating Scaling Laws in Transformers and MLPs on Random Sequence Memorization Tasks**  
By Niranjan Vijaya Krishnan, Christine Guo, Diya Hundiwala  
📄 [Read the full paper here](https://niruvk.github.io/Transformers_Research/Transformers_Research.pdf)

---

## 🔍 Overview

This project explores **memorization thresholds** in neural networks—how well models like MLPs and decoder-only Transformers can **memorize random sequence mappings**, under a fixed parameter budget. Our experiments probe how different architectural features (depth, width, attention heads) impact memorization on synthetic tasks.

---

## ✨ Key Contributions

- 🔍 **Comparative Analysis** of MLPs vs. Transformers on random bijective sequence mapping tasks.
- 📈 **Scaling Law Insights**: MLPs benefit from width, while Transformers benefit from multi-head attention.
- 🧪 **Controlled Experiments**: All models are constrained to identical parameter budgets (10k–128k) for fair comparison.
- 📊 **Memorization Threshold**: Defined as the maximum number of mappings a model can learn perfectly.

---

## 📁 Project Structure

```
Transformers_Research/
│
├── mlp_experiments/                 # MLP model architecture + training scripts
│   ├── train_mlp.py
│   └── mlp_utils.py
│
├── transformer_experiments/        # Transformer model + training code
│   ├── train_transformer.py
│   └── transformer_utils.py
│
├── dataset/
│   └── synthetic_generator.py       # Generates random bijective sequence mappings
│
├── threshold_search/
│   └── binary_search.py             # Logic for finding memorization threshold
│
├── visualizations/
│   └── plots.ipynb                  # Graphs for performance vs. depth/head count
│
├── results/
│   ├── mlp_results.csv
│   ├── transformer_results.csv
│   └── figures/
│
├── requirements.txt
├── README.md
└── Transformers_Research.pdf        # Full paper (results + theory)
```

---

## 🧪 Methodology Summary

### Task

Models are trained to memorize **random bijective mappings** between fixed-length sequences of tokens (length = 10, vocab size = 10). No generalization is expected—the goal is pure memorization.

### Model Architectures

- **MLPs**: Fully connected feedforward networks used for sequence-to-sequence learning.
- **Decoder-Only Transformers**: Used for autoregressive token prediction.

Each model configuration is evaluated on:
- Varying depth vs. width (MLPs)
- Attention heads vs. feedforward size (Transformers)

### Memorization Threshold Search

- We use **binary search** to find the highest dataset size where the model achieves **100% accuracy**.
- Training stops when:
  - Accuracy < 100%
  - Or training plateaus for 3 epochs

---

## 📊 Key Results

### MLP Insights

| Parameter Count | Best Depth | Max Threshold |
|-----------------|------------|----------------|
| 10,000          | 1 layer    | 467 sequences  |
| 30,000          | 1 layer    | 1728 sequences |
| 90,000          | 1 layer    | 3648 sequences |

✅ Wider and shallower MLPs memorize better  
⚠️ Deep MLPs collapse even with more parameters

---

### Transformer Insights

| Param Count | Max Heads | Max Threshold |
|-------------|-----------|----------------|
| 64,000      | 32        | 77 sequences   |
| 128,000     | 64        | 42 sequences   |

✅ Attention heads improve memorization  
📉 Depth mildly reduces performance  
❌ Transformers less efficient than MLPs for pure memorization

---

## ⚙️ Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

#### MLP

```bash
python mlp_experiments/train_mlp.py --params 10000 --layers 1
```

#### Transformer

```bash
python transformer_experiments/train_transformer.py --params 64000 --heads 32 --layers 2
```

### 3. Find Memorization Threshold

```bash
python threshold_search/binary_search.py --model mlp --params 30000
```

---

## 📚 Citation

```bibtex
@article{krishnan2024memorization,
  title={The Memorization Capabilities of Neural Networks: Understanding Scaling Laws in Autoregressive and Seq2Seq Learning},
  author={Krishnan, Niranjan Vijaya and Guo, Christine and Hundiwala, Diya},
  journal={GitHub},
  year={2024},
  url={https://github.com/niruvk/Transformers_Research}
}
```
