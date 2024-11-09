# StableBoothRT


Install:

```pip3 install torch torchvision torchaudio```

or (preferrably to match dependencies in requirements.txt)

```pip install torch==2.5.0 torchaudio==2.5.0 torchvision==0.20.0```

then 

```pip install -r requirements.txt```

Windows:

for Cuda 12.1
```pip3 install torch==2.5.0 torchaudio==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121```

if you had already installed these before, use force-reinstall argument.
```pip3 install --force-reinstall torch==2.5.0 torchaudio==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121```

