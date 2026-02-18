# learn-nation-phoneme-aligner
Python code to fetch, test, split/parse individual phoneme audio from 11labs

# Running

Activate the conda env, then run:

```bash
conda activate mfa
python align_dataset.py
```

**New terminal / Cursor restart:** Conda does not auto-activate. Either run `conda activate mfa` in the terminal before running, or in Cursor: **Cmd+Shift+P → "Python: Select Interpreter"** and choose the `mfa` conda env—then enable **Python › Terminal: Activate Environment** in settings so new terminals auto-activate that env.

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
