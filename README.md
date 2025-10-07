# 🏗️ Loom-Train  
*A Simple & Efficient Training Framework for Long-Context LLMs*

> Optimized for scalability, memory efficiency, and seamless integration — built to unlock the full potential of long-context large language models.

---

## 📅 Update Log

- **📅 2025-10-07** — 🚀 **Initial Release**: Loom-Train is now live!  
  ✅ Native support for [🤗 Hugging Face Trainer](https://github.com/huggingface/transformers)  
  ✅ Optimized attention with [🌀 Ring-Flash-Attention](https://github.com/zhuzilin/ring-flash-attention)  
  ✅ Lightweight, plug-and-play design for long-sequence training (128K+ tokens)

---

## ✨ Key Features

- 🔧 **Plug-and-Play**: Drop-in replacement for HF Trainer — no major code changes needed.  
- 🚀 **Memory-Efficient**: Leverages Ring-Flash-Attention to reduce GPU memory footprint by up to 50%.  
- 📈 **Scalable**: Seamlessly scales to 100K+ context lengths without sacrificing speed.  
- ⚡ **Fast Setup**: Minimal dependencies, easy installation via `pip install loom-train`.

---

## 💻 Environment & Installation

To install the`loomtrain` package from the gitee repository, run:

```bash
git clone https://github.com/iiGray/loomtrain.git
cd loomtrain
conda create -n lcf python=3.10 -y
conda activate lcf
pip install -e .
# install flash attention
Download the suitable version of flash_attn from https://github.com/Dao-AILab/flash-attention/releases
pip install <path_to_flash_attn_whl_file>
pip install ring_flash_attn
```

---

## 🛠️ Getting Started

```bash
pip install loom-train
```

Then just swap your `Trainer` with `LoomTrainer`:

```python
from loom_train import LoomTrainer

trainer = LoomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # ... rest unchanged!
)
```

---

## 🤝 Contributing

We welcome contributions! Whether it’s bug fixes, new features, or documentation improvements — feel free to open an issue or PR.  
Let’s build the future of long-context training, together. 💪

---

## 📬 Contact

Questions? Suggestions? Reach out at: `iiiigray19@gmail.com` and `zctang2000@gmail.com`
