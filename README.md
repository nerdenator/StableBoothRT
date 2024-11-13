# StableBoothRT

## 1) Download Models using Git LFS: https://git-lfs.com/

   - Dreamshaper v7 LCM: https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7
   - SDXL Turbo: https://huggingface.co/stabilityai/sdxl-turbo

```
git clone https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7
```


```
git clone https://huggingface.co/stabilityai/sdxl-turbo
```

## 3) create python environment with Python 3.10.11 and activate it

MacOS: https://www.python.org/downloads/macos/
Windows: https://www.python.org/downloads/windows/

MacOS:

find your python versions in terminal:
```
ls /usr/local/bin/python*
```

then use the versions to create a python venv
```
{path here} -m venv SBRT_venv
```

activate the environment

```
source SBRT_venv/bin/activate
```

Windows:

find your python versions in cmd:
```
py -0p
```

then use the versions to create a python venv
```
{path here} -m venv SBRT_venv
```

activate the environment

```
SBRT_venv\Scripts\activate
```

## 3) clone repo and cd into it

```
git clone https://github.com/rmasiso/StableBoothRT
```

```
cd StableBoothRT
```

## 4) Install Pytorch and install requirements.txt file

MacOS:

Install: 

```
pip3 install torch torchvision torchaudio
```

or (preferrably to match dependencies in requirements.txt)

```
pip install torch==2.5.0 torchaudio==2.5.0 torchvision==0.20.0
```

then 

```
pip install -r requirements.txt
```

Windows:

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

