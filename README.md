# StableBoothRT

Notes: 
- [additional instructions for how to install additional models will come soon]

## 1) Download Models using Git LFS: https://git-lfs.com/

   - Dreamshaper v7 LCM: https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7
   - SDXL Turbo: https://huggingface.co/stabilityai/sdxl-turbo
   - ControlNet for Canny Edge Detection: https://huggingface.co/lllyasviel/control_v11p_sd15_canny

```
git clone https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7
```


```
git clone https://huggingface.co/stabilityai/sdxl-turbo
```

```
git clone https://huggingface.co/lllyasviel/control_v11p_sd15_canny
```

## 2) Install uv
*note: these are instructions from https://docs.astral.sh/uv/getting-started/installation/*
For MacOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

## 3) Clone repository & change into directory
```
git clone https://github.com/rmasiso/StableBoothRT

cd StableBoothRT
```

## 4) Set up project using uv:
```
uv venv --python 3.10.11
source .venv/bin/activate
uv sync
```


## 5) clone repo and cd into it

```
git clone https://github.com/rmasiso/StableBoothRT
```

```
cd StableBoothRT
```

## 6) FOR WINDOWS: Install Pytorch and install requirements.txt file

for Cuda 12.1
```
pip3 install torch==2.5.0 torchaudio==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121
```

if you had already installed these before, use force-reinstall argument.
```
pip3 install --force-reinstall torch==2.5.0 torchaudio==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121
```

## 5) Run!

```python app.py```

