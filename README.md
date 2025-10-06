# Loom-Train
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
