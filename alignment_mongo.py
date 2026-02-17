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
from typing import Any, List, Optional, Union

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from bson import ObjectId


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def get_client(uri: str = "mongodb://localhost:27017", **kwargs) -> MongoClient:
    """Create a MongoDB client."""
    return MongoClient(uri, **kwargs)


def get_collection(
    uri: str = "mongodb://localhost:27017",
    database: str = "learn_nation",
    collection: str = "alignment",
    **client_kwargs,
) -> Collection:
    """Return the alignment collection."""
    client = get_client(uri, **client_kwargs)
    db: Database = client[database]
    return db[collection]


# ---------------------------------------------------------------------------
# Read (same data shape as dataset/alignment.json)
# ---------------------------------------------------------------------------


def read_alignment_entries(
    collection: Optional[Collection] = None,
    uri: Optional[str] = None,
    database: Optional[str] = None,
    coll_name: Optional[str] = None,
    created_before: Optional[datetime] = None,
    **client_kwargs,
) -> List[dict]:
    """
    Read documents from the alignment collection, optionally filtered by created timestamp.

    Returns a list of documents with the same logical shape as dataset/alignment.json
    (e.g. script, voiceId, voiceKeyHash, audioBase64, alignment, normalisedAlignment, etc.).

    created_after: only include documents whose `created_field` is >= this (timezone-aware).
    created_before: only include documents whose `created_field` is < this (timezone-aware).
    created_field: name of the document field used for the timestamp (default "created").
    """
    if collection is None:
        if uri is None or database is None or coll_name is None:
            raise ValueError("Either pass collection= or (uri=, database=, coll_name=).")
        collection = get_collection(uri=uri, database=database, collection=coll_name, **client_kwargs)

    query: dict = {}
    if created_before is not None:
        query["created"] = {"$lt": created_before}

    cursor = collection.find(query)
    return list(cursor)


def read_alignment_entry_by_voice_key_hash(
    voice_key_hash: str,
    collection: Optional[Collection] = None,
    uri: Optional[str] = None,
    database: Optional[str] = None,
    coll_name: Optional[str] = None,
    **client_kwargs,
) -> Optional[dict]:
    """Return a single document matching voiceKeyHash, or None."""
    if collection is None:
        if uri is None or database is None or coll_name is None:
            raise ValueError("Either pass collection= or (uri=, database=, coll_name=).")
        collection = get_collection(uri=uri, database=database, collection=coll_name, **client_kwargs)

    return collection.find_one({"voiceKeyHash": voice_key_hash})


def read_alignment_entry_by_id(
    document_id: Union[str, ObjectId],
    collection: Optional[Collection] = None,
    uri: Optional[str] = None,
    database: Optional[str] = None,
    coll_name: Optional[str] = None,
    **client_kwargs,
) -> Optional[dict]:
    """Return a single document by _id, or None."""
    if collection is None:
        if uri is None or database is None or coll_name is None:
            raise ValueError("Either pass collection= or (uri=, database=, coll_name=).")
        collection = get_collection(uri=uri, database=database, collection=coll_name, **client_kwargs)

    oid = ObjectId(document_id) if isinstance(document_id, str) else document_id
    return collection.find_one({"_id": oid})


# ---------------------------------------------------------------------------
# Phoneme alignment schema
# ---------------------------------------------------------------------------

# Each item in phonemes.alignment:
#   cmu: str
#   start: float
#   end: float


def write_phonemes_to_document(
    alignment: List[dict],
    collection: Optional[Collection] = None,
    document_id: Optional[Union[str, ObjectId]] = None,
    voice_key_hash: Optional[str] = None,
    uri: Optional[str] = None,
    database: Optional[str] = None,
    coll_name: Optional[str] = None,
    **client_kwargs,
) -> Optional[Any]:
    """
    Write the exported phoneme alignment back to the same document in the collection.

    Schema written:
      phonemes: {
        created: <datetime>,
        alignment: [ { cmu: str, start: number, end: number }, ... ]
      }
    All other document fields are left unchanged.

    Provide exactly one of: collection, or (uri, database, coll_name).
    Provide exactly one of: document_id, or voice_key_hash (to identify the document).

    alignment: list of dicts with keys "cmu", "start", "end" (e.g. from export JSON or
               align_dataset.postprocess + mapping to cmu/start/end).

    Returns the result of update_one (or None if no collection/doc identified).
    """
    if collection is None:
        if uri is None or database is None or coll_name is None:
            raise ValueError("Either pass collection= or (uri=, database=, coll_name=).")
        collection = get_collection(uri=uri, database=database, collection=coll_name, **client_kwargs)

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


def load_phonemes_from_export_json(file_path: str) -> List[dict]:
    """
    Load an alignment list from an exported JSON file (e.g. dataset/phonemes_json/<hash>.json).
    Returns a list of { "cmu", "start", "end" } dicts suitable for write_phonemes_to_document.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [{"cmu": x["cmu"], "start": x["start"], "end": x["end"]} for x in data]
