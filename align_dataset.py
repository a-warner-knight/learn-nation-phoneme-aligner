import base64
import io
import json
import os
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
    """Convert audio from dataset/alignment.json (audioBase64) to 16k mono wav for MFA.
    Uses voiceKeyHash as the output filename."""
    alignment_path = DATASET / "alignment.json"
    with open(alignment_path, "r", encoding="utf8") as f:
        entries = json.load(f)

    for entry in entries:
        voice_key_hash = entry.get("voiceKeyHash")
        audio_b64 = entry.get("audioBase64")
        if not voice_key_hash or not audio_b64:
            continue

        wav_path = WAV_DIR / f"{voice_key_hash}.wav"
        if wav_path.exists():
            continue

        audio_bytes = base64.b64decode(audio_b64)
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(wav_path, format="wav")

        print("Converted:", voice_key_hash)


def copy_transcripts():
    """Build .lab files from dataset/alignment.json: join normalisedAlignment.characters, named by voiceKeyHash."""
    alignment_path = DATASET / "alignment.json"
    with open(alignment_path, "r", encoding="utf8") as f:
        entries = json.load(f)

    for entry in entries:
        voice_key_hash = entry.get("voiceKeyHash")
        norm = entry.get("normalisedAlignment") or {}
        characters = norm.get("characters")
        if not voice_key_hash or characters is None:
            continue

        text = "".join(characters).strip()
        lab_path = LAB_DIR / f"{voice_key_hash}.lab"
        with open(lab_path, "w", encoding="utf8") as f:
            f.write(text)

        print("Wrote lab:", voice_key_hash)


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
    # ensure_dirs()
    # convert_audio()
    # copy_transcripts()
    run_mfa()
    parse_outputs()
    print("Done.")


if __name__ == "__main__":
    main()
