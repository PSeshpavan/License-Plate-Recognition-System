import os
from typing import Any
from bson import ObjectId
from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv
load_dotenv()

# ── Mongo connection settings ──────────────────────────────────────────────
def mongo_connection():
    """Initialise MongoDB connection and return (client, db, fs)."""
    uri  = os.getenv("MONGO_URI")
    name = os.getenv("DATABASE_NAME")
    if not (uri and name):
        raise ValueError("MONGO_URI and DATABASE_NAME must be set.")
    client = MongoClient(uri)
    db     = client[name]
    fs     = gridfs.GridFS(db)
    return client, db, fs

# **Create the globals once**
_client, _db, _fs = mongo_connection()
# (keep the leading underscore to mark “private module globals”)

# ── Public helpers ─────────────────────────────────────────────────────────
def put_file(data: bytes, *, filename: str, content_type: str, **metadata: Any) -> str:
    """Store data in GridFS and return the file’s ObjectId as str."""
    return str(_fs.put(
        data, filename=filename, content_type=content_type, metadata=metadata
    ))

def get_bytes(file_id: str) -> bytes:
    """Return raw bytes for *file_id* from GridFS."""
    return _fs.get(ObjectId(file_id)).read()

def get_gridout(file_id: str):
    """Return the GridOut object (handy for Flask Response)."""
    return _fs.get(ObjectId(file_id))

# ── DB-size guard ──────────────────────────────────────────────────────────
THRESHOLD_BYTES = int(os.getenv("MONGO_THRESHOLD_BYTES", 300 * 1024 * 1024))

def vacate_if_low_space() -> bool:
    """If DB ≥ threshold, delete *only* GridFS data and return True."""
    try:
        used = _db.command("dbstats").get("storageSize", 0)
        if used >= THRESHOLD_BYTES:
            print(f"[INFO] DB usage {used/1e6:.1f} MB ≥ limit – purging GridFS")
            _db.fs.files.delete_many({})
            _db.fs.chunks.delete_many({})
            return True
    except Exception as exc:
        print(f"[WARN] Could not check db size: {exc}")
    return False

__all__ = [
    "put_file", "get_bytes", "get_gridout",
    "mongo_connection", "vacate_if_low_space"   # ← export it so app.py can import
]
