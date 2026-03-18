# learn-nation-phoneme-aligner
Python code to fetch, test, split/parse individual phoneme audio from 11labs

# Running

Activate the conda env, then run:

```bash
conda activate mfa
python align_dataset.py
```

NB See 'Command Line Arguments' section below.

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

# Command Line Arguments

To update entries in mongo, use the `--use-mongo <mongourl>` option with a path to the mongo instance.
This will read the .audioBase64 field from every document in the `audioentries` collection, write them to disk locally, run MFA on them all, the update each mongo audioentries document with a .phonemes field with {created: Date, alignment: {cmu, start, end}[]}.
To use CMU phonemes (AH, AY, etc) instead of default IPA, use the `--cmu` option.
For LN, we currently use CMU.

# Learn Nation Example Command

An example command to directly update the prod instance would be

```
python align_dataset.py --use-mongo "mongodb://[user]:[password]@170.64.231.205:27017/learn-nation?authSource=admin" --cmu
```