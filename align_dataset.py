import base64
import io
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import List
from pydub import AudioSegment
from praatio import textgrid
import mongo_adaptor

# ---------------- CONFIG ----------------

DATASET = Path("dataset")
AUDIO_IN = DATASET / "audio"
TEXT_IN = DATASET / "transcripts"

WORK_DIR = Path("mfa_work")
WAV_DIR = WORK_DIR
LAB_DIR = WORK_DIR
ALIGN_DIR = WORK_DIR / "aligned"
JSON_OUT = DATASET / "phonemes_json"

MIN_PHONE_DUR = 0.035      # 35 ms
MERGE_THRESHOLD = 0.025    # merge phones shorter than this
ANTICIPATION_SHIFT = 0.015 # shift starts earlier for animation

# Read mongo config from env
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DATABASE = os.environ.get("MONGO_DATABASE", "learn_nation")
MONGO_COLLECTION = os.environ.get("MONGO_COLLECTION", "audioentries")
# ----------------------------------------

def ensure_dirs():
    for d in [WAV_DIR, LAB_DIR, ALIGN_DIR, JSON_OUT]:
        d.mkdir(parents=True, exist_ok=True)

def get_entries(from_mongo=False):
    if from_mongo:
        return mongo_adaptor.read_alignment_entries(uri=MONGO_URI, database=MONGO_DATABASE, collection=MONGO_COLLECTION)
    else:
        alignment_path = DATASET / "alignment.json"
        with open(alignment_path, "r", encoding="utf8") as f:
            return json.load(f)

def convert_audio(entries: List[dict]):
    """Convert audio from dataset/alignment.json (audioBase64) to 16k mono wav for MFA.
    Uses voiceKeyHash as the output filename."""

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


def copy_transcripts(entries: List[dict]):
    """Build .lab files from dataset/alignment.json: join normalisedAlignment.characters, named by voiceKeyHash."""
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


def run_mfa(acoustic_model: str, dictionary: str):
    """Run MFA alignment"""
    print("Running MFA alignment...")
    work_dir = str(WORK_DIR.resolve())
    align_dir = str(ALIGN_DIR.resolve())
    print(f"Command: mfa align {work_dir} {dictionary} {acoustic_model} {align_dir} --clean --overwrite")
    
    subprocess.run([
        "mfa",
        "align",
        work_dir,
        dictionary,
        acoustic_model,
        align_dir,
        "--clean",
        "--overwrite"
    ], check=True)


def load_phones_from_textgrid(tg_path):
    """Extract phoneme intervals from MFA TextGrid"""
    tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=False)

    # MFA typically names tier "phones"
    phone_tier = tg._tierDict.get("phones")
    if phone_tier is None:
        raise RuntimeError(f"No phones tier in {tg_path}")

    phones = []
    for start, end, label in phone_tier.entries:
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

    # Append to original alignment.json under phonemeAlignment
    alignment_path = DATASET / "alignment.json"
    with open(alignment_path, "r", encoding="utf8") as f:
        entries = json.load(f)
    for entry in entries:
        if entry.get("voiceKeyHash") == name:
            entry["phonemeAlignment"] = {
                "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "alignment": out,
            }
            break
    with open(alignment_path, "w", encoding="utf8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def parse_outputs(write_to_mongo=False):
    """Parse MFA TextGrids and publish"""

    for tg_file in ALIGN_DIR.glob("**/*.TextGrid"):
        name = tg_file.stem

        # Parse TextGrid into phones
        phones = load_phones_from_textgrid(tg_file)

        # Postprocess phones (anticipation shift, merge tiny phones, enforce minimum duration)
        phones = postprocess(phones)

        # Write to individual JSON files, append/overwrite field to object in alignment.json
        export_json(name, phones)

        if write_to_mongo:
            mongo_adaptor.write_phonemes_to_document(phones, uri=MONGO_URI, database=MONGO_DATABASE, collection=MONGO_COLLECTION)

        print("Exported:", name)


def main(migrate_mongo=False, phone_type="cmu"):
    # Set phone type
    if (phone_type == "cmu"):
        acoustic_model = "english_us_arpa"
        dictionary = "english_us_arpa"
    else:
        acoustic_model = "english_mfa"
        dictionary = "english_mfa"

    if migrate_mongo:
        print("Processing audio entries from MongoDB.")
        mongo_adaptor.init(uri=MONGO_URI, database=MONGO_DATABASE, collection=MONGO_COLLECTION)

    ensure_dirs()
    entries = get_entries(migrate_mongo)
    convert_audio(entries)
    copy_transcripts(entries)
    run_mfa(acoustic_model, dictionary)
    parse_outputs(migrate_mongo)

    print("Done.")


if __name__ == "__main__":
    # Check for '--migrate-mongo' flag
    migrate_mongo = True if "--use-mongo" in sys.argv[1:] else False
    phone_type = "cmu" if "--cmu" in sys.argv[1:] else "ipa"
    main(migrate_mongo, phone_type)
