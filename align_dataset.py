import base64
import io
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import List
from pydub import AudioSegment
from praatio import textgrid
import mongo_adaptor
from mongo_adaptor import PhonemeSegment

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
MONGO_DATABASE = os.environ.get("MONGO_DATABASE", "learn-nation")
MONGO_COLLECTION = os.environ.get("MONGO_COLLECTION", "audioentries")
# ----------------------------------------

def ensure_dirs(clean=False):
    if clean:
        for d in [WORK_DIR]:
            if d.exists():
                shutil.rmtree(d)

    for d in [WAV_DIR, LAB_DIR, ALIGN_DIR, JSON_OUT]:
        d.mkdir(parents=True, exist_ok=True)

def get_entries(from_mongo=False):
    if from_mongo:
        return mongo_adaptor.read_alignment_entries()
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


def postprocess(phones: List[dict], is_cmu: bool):

    # ---- additional schwa after plosive ends of words ----
    PLOSIVES = {"B", "D", "G", "P", "T", "K"}
    SCHWA = "EH" if is_cmu else "ɛ"
    with_schwas = [] # build a new list of phones with schwa's inserted if necesssary

    for i, p in enumerate(phones):
        with_schwas.append(p) # add the current phone to the result

        # check if plosive
        if p["phone"] not in PLOSIVES:
            continue
        
        # determine next phone + gap
        if i + 1 < len(phones):
            next_p = phones[i+1]
            gap = next_p["start"] - p["end"]
        else:
            gap = float('inf') # last phoneme, so there is no gap

        if gap > MIN_PHONE_DUR:
            with_schwas.append({
                "phone": SCHWA,
                "start": p["end"],
                "end": p["end"] + MIN_PHONE_DUR
                })

    # ---- anticipation shift ----
    for p in with_schwas:
        p["start"] = max(0.0, p["start"] - ANTICIPATION_SHIFT)

    # ---- merge tiny phones ----
    merged = []
    for p in with_schwas:
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


def export_json(name: str, phones: List[PhonemeSegment]):

    with open(JSON_OUT / f"{name}.json", "w") as f:
        json.dump(phones, f, indent=2)

    # Append to original alignment.json under phonemeAlignment
    alignment_path = DATASET / "alignment.json"
    with open(alignment_path, "r", encoding="utf8") as f:
        entries = json.load(f)
    for entry in entries:
        if entry.get("voiceKeyHash") == name:
            entry["phonemeAlignment"] = {
                "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "alignment": phones,
            }
            break
    with open(alignment_path, "w", encoding="utf8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def parse_outputs(write_to_mongo=False, is_cmu=True):
    """Parse MFA TextGrids and publish"""

    for tg_file in ALIGN_DIR.glob("**/*.TextGrid"):
        name = tg_file.stem

        # Parse TextGrid into phones
        phones = load_phones_from_textgrid(tg_file)

        # Postprocess phones (anticipation shift, merge tiny phones, enforce minimum duration)
        phones = postprocess(phones, is_cmu)

        # Cleanup (round) times, map to cmu/start/end
        alignment = [PhonemeSegment(cmu=p["phone"], start=round(p["start"], 4), end=round(p["end"], 4)) for p in phones]

        # Write to individual JSON files, append/overwrite field to object in alignment.json
        export_json(name, alignment)

        if write_to_mongo:
            mongo_adaptor.write_phonemes_to_document(alignment, voice_key_hash=name)

        print("Exported:", name)


def main(migrate_mongo=False, is_cmu=True):
    # Set phone type
    acoustic_model = "english_us_arpa" if is_cmu else "english_mfa"
    dictionary = "english_us_arpa" if is_cmu else "english_mfa"

    if migrate_mongo:
        print("Processing audio entries from MongoDB.")
        mongo_adaptor.init(uri=MONGO_URI, database=MONGO_DATABASE, collection=MONGO_COLLECTION)

    ensure_dirs(clean=True)
    entries = get_entries(migrate_mongo)
    convert_audio(entries)
    copy_transcripts(entries)
    run_mfa(acoustic_model, dictionary)
    parse_outputs(migrate_mongo, is_cmu)

    if migrate_mongo:
        mongo_adaptor.close()
        
    print("Done.")


if __name__ == "__main__":
    argv = sys.argv[1:]
    migrate_mongo = "--use-mongo" in argv
    if migrate_mongo:
        i = argv.index("--use-mongo")
        if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
            sys.stderr.write("Error: --use-mongo requires a URI (e.g. --use-mongo mongodb://localhost:27017)\n")
            sys.exit(1)
        MONGO_URI = argv[i + 1]
    is_cmu = "--cmu" in argv
    main(migrate_mongo, is_cmu)
