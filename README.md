# learn-nation-phoneme-aligner
Python code to fetch, test, split/parse individual phoneme audio from 11labs

# Conda

Initial setup was with:

```
conda create -n mfa -c conda-forge montreal-forced-aligner python=3.10
conda activate mfa
pip install praatio pydub

mfa model download acoustic english_mfa
mfa model download dictionary english_us_arpa
```

Confirm mfa model downloads with:

```
mfa model list dictionary
mfa model list acoustic
```
