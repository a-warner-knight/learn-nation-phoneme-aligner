import os
import json
import subprocess
from pathlib import Path
from pydub import AudioSegment
from praatio import textgrid

# ---------------- CONFIG ----------------

DATASET = Path("dataset")
AUDIO_IN = DATASET / "audio"
TEXT_IN = DATASET / "transcripts"

WORK_DIR = Path("mfa_work")
WAV_DIR = WORK_DIR / "wav"
LAB_DIR = WORK_DIR / "lab"
ALIGN_DIR = WORK_DIR / "aligned"
JSON_OUT = DATASET / "phonemes_json"

ACOUSTIC_MODEL = "english_mfa"
DICTIONARY = "english_us_arpa"

MIN_PHONE_DUR = 0.035      # 35 ms
MERGE_THRESHOLD = 0.025    # merge phones shorter than this
ANTICIPATION_SHIFT = 0.015 # shift starts earlier for animation

# ----------------------------------------

def ensure_dirs():
    for d in [WAV_DIR, LAB_DIR, ALIGN_DIR, JSON_OUT]:
        d.mkdir(parents=True, exist_ok=True)


def convert_audio():
    """Convert all audio to 16k mono wav for MFA"""
    for audio_file in AUDIO_IN.iterdir():
        if audio_file.suffix.lower() not in [".wav", ".mp3", ".flac", ".m4a"]:
            continue

        wav_path = WAV_DIR / (audio_file.stem + ".wav")

        if wav_path.exists():
            continue

        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(wav_path, format="wav")

        print("Converted:", audio_file.name)


def copy_transcripts():
    """Convert txt -> lab files MFA expects"""
    for txt_file in TEXT_IN.glob("*.txt"):
        lab_path = LAB_DIR / (txt_file.stem + ".lab")
        with open(txt_file, "r", encoding="utf8") as f:
            text = f.read().strip()

        with open(lab_path, "w", encoding="utf8") as f:
            f.write(text)


def run_mfa():
    """Run MFA alignment"""
    print("Running MFA alignment...")

    subprocess.run([
        "mfa",
        "align",
        str(WORK_DIR),
        DICTIONARY,
        ACOUSTIC_MODEL,
        str(ALIGN_DIR),
        "--clean",
        "--overwrite"
    ], check=True)


def load_phones_from_textgrid(tg_path):
    """Extract phoneme intervals from MFA TextGrid"""
    tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=False)

    # MFA typically names tier "phones"
    phone_tier = tg.tierDict.get("phones")
    if phone_tier is None:
        raise RuntimeError(f"No phones tier in {tg_path}")

    phones = []
    for start, end, label in phone_tier.entryList:
        label = label.strip()
        if not label or label.lower() == "spn":
            continue

        phones.append({
            "phone": label.upper(),
            "start": float(start),
            "end": float(end),
        })

    return phones


def postprocess(phones):

    # ---- anticipation shift ----
    for p in phones:
        p["start"] = max(0.0, p["start"] - ANTICIPATION_SHIFT)

    # ---- merge tiny phones ----
    merged = []
    for p in phones:
        dur = p["end"] - p["start"]

        if merged and dur < MERGE_THRESHOLD:
            # merge into previous phone
            merged[-1]["end"] = p["end"]
        else:
            merged.append(p)

    # ---- enforce minimum duration ----
    for i, p in enumerate(merged):
        dur = p["end"] - p["start"]
        if dur < MIN_PHONE_DUR:
            deficit = MIN_PHONE_DUR - dur
            p["end"] += deficit

            # push next start forward if overlapping
            if i+1 < len(merged) and merged[i+1]["start"] < p["end"]:
                merged[i+1]["start"] = p["end"]

    return merged


def export_json(name, phones):

    out = []

    for p in phones:
        out.append({
            "cmu": p["phone"],   # CMU ARPABET
            "start": round(p["start"], 4),
            "end": round(p["end"], 4)
        })

    with open(JSON_OUT / f"{name}.json", "w") as f:
        json.dump(out, f, indent=2)


def parse_outputs():
    """Parse MFA TextGrids into JSON"""

    for tg_file in ALIGN_DIR.glob("**/*.TextGrid"):
        name = tg_file.stem

        phones = load_phones_from_textgrid(tg_file)
        phones = postprocess(phones)
        export_json(name, phones)

        print("Exported:", name)


def main():
    ensure_dirs()
    convert_audio()
    copy_transcripts()
    run_mfa()
    parse_outputs()
    print("Done.")


if __name__ == "__main__":
    main()
