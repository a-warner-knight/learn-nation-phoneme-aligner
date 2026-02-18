"""
Read alignment data from a MongoDB collection (same shape as dataset/alignment.json)
and write phoneme alignment results back to documents.

Batch workflow (e.g. 1500 docs, or 10–100 with created_before filter):
  1. read_alignment_entries(..., created_before=...) → list of docs
  2. For each doc: decode audioBase64 → write {voiceKeyHash}.wav; build transcript → write {voiceKeyHash}.lab
  3. Run MFA once on the whole wav+lab directory (one align command for all)
  4. For each generated TextGrid: load_phones_from_textgrid → postprocess → write_phonemes_to_document(..., voice_key_hash=...)
  This keeps one Mongo read, one MFA run, and N writes back to Mongo.
"""

import json
from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from bson import ObjectId
from typing import Any, List, Optional, TypedDict, Union


class PhonemeSegment(TypedDict):
    """One segment in phonemes.alignment: cmu (ARPABET label), start/end times in seconds."""
    cmu: str
    start: float
    end: float


# ---------------------------------------------------------------------------
# Connection / session config
# ---------------------------------------------------------------------------

# Stored after init(); used by read_* and write_phonemes_to_document. Call init() before any other function.
_config: dict = {
    "uri": None,
    "database": None,
    "coll_name": None,
    "client_kwargs": None,
    "client": None,
    "collection": None,
}


def init(
    uri: str = "mongodb://localhost:27017",
    database: str = "learn-nation",
    collection: str = "audioentries",
    **client_kwargs,
) -> None:
    """
    Set the connection config for this module. Must be called once before using read_alignment_entries(),
    write_phonemes_to_document(), etc.
    """
    client = MongoClient(uri, **client_kwargs)
    db: Database = client[database]
    coll = db[collection]
    _config["uri"] = uri
    _config["database"] = database
    _config["coll_name"] = collection
    _config["client_kwargs"] = client_kwargs
    _config["client"] = client
    _config["collection"] = coll


def close() -> None:
    """
    Close the MongoDB client created by init(). Idempotent; safe to call multiple times.
    Called automatically on process exit if init() was used. Can also be called explicitly
    (e.g. from main() in a try/finally) to release the connection early.
    """
    client = _config.get("client")
    if client is not None:
        client.close()
        _config["client"] = None
        _config["collection"] = None


def _get_collection() -> Collection:
    """Return the collection set by init(). Raises if init() has not been called."""
    if _config["collection"] is None:
        raise ValueError("Call mongo_adaptor.init(uri=..., database=..., collection=...) first.")
    return _config["collection"]


# ---------------------------------------------------------------------------
# Read (same data shape as dataset/alignment.json)
# ---------------------------------------------------------------------------


def read_alignment_entries(
    created_before: Optional[datetime] = None,
) -> List[dict]:
    """
    Read documents from the alignment collection, optionally filtered by created timestamp.
    Requires init() to have been called.

    Returns a list of documents with the same logical shape as dataset/alignment.json
    (e.g. script, voiceId, voiceKeyHash, audioBase64, alignment, normalisedAlignment, etc.).

    created_before: only include documents whose `created` field is < this (timezone-aware).
    """
    collection = _get_collection()

    query: dict = {}
    if created_before is not None:
        query["created"] = {"$lt": created_before}

    cursor = collection.find(query)
    return list(cursor)


def read_alignment_entry_by_voice_key_hash(voice_key_hash: str) -> Optional[dict]:
    """Return a single document matching voiceKeyHash, or None. Requires init() to have been called."""
    return _get_collection().find_one({"voiceKeyHash": voice_key_hash})


def read_alignment_entry_by_id(document_id: Union[str, ObjectId]) -> Optional[dict]:
    """Return a single document by _id, or None. Requires init() to have been called."""
    oid = ObjectId(document_id) if isinstance(document_id, str) else document_id
    return _get_collection().find_one({"_id": oid})


# ---------------------------------------------------------------------------
# Phoneme alignment schema (phonemes.alignment: List[PhonemeSegment])
# ---------------------------------------------------------------------------


def write_phonemes_to_document(
    alignment: List[PhonemeSegment],
    document_id: Optional[Union[str, ObjectId]] = None,
    voice_key_hash: Optional[str] = None,
) -> Optional[Any]:
    """
    Write phoneme alignment back to the same document in the collection.
    Requires init() to have been called.

    Schema written:
      phonemes: {
        created: <datetime>,
        alignment: [ { cmu: str, start: number, end: number }, ... ]
      }
    All other document fields are left unchanged.

    Provide exactly one of: document_id, or voice_key_hash (to identify the document).

    alignment: list of PhonemeSegment (cmu, start, end).

    Returns the result of update_one.
    """
    collection = _get_collection()

    if (document_id is None) == (voice_key_hash is None):
        raise ValueError("Provide exactly one of document_id= or voice_key_hash=.")

    if document_id is not None:
        oid = ObjectId(document_id) if isinstance(document_id, str) else document_id
        filter_query = {"_id": oid}
    else:
        filter_query = {"voiceKeyHash": voice_key_hash}

    # Normalise alignment items to exactly { cmu, start, end }
    alignment_payload = [
        {
            "cmu": item["cmu"],
            "start": float(item["start"]),
            "end": float(item["end"]),
        }
        for item in alignment
    ]

    phonemes = {
        "created": datetime.now(timezone.utc),
        "alignment": alignment_payload,
    }

    result = collection.update_one(filter_query, {"$set": {"phonemes": phonemes}})
    return result

