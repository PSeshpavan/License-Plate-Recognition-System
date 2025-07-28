import os
from typing import Any

from bson import ObjectId
from pymongo import MongoClient
import gridfs
import dotenv
dotenv.load_dotenv()



# ── Mongo connection settings ──────────────────────────────────────────────
def mongo_connection():
    """Initialize MongoDB connection."""
    mongo_uri = os.getenv("MONGO_URI")
    database_name = os.getenv("DATABASE_NAME")
    
    if not mongo_uri or not database_name:
        raise ValueError("MONGO_URI and DATABASE_NAME must be set in environment variables.")
    
    client = MongoClient(mongo_uri)
    db = client[database_name]
    fs = gridfs.GridFS(db)
    
    return client, db, fs


# ── Public helpers ─────────────────────────────────────────────────────────
def put_file(data: bytes, *, filename: str, content_type: str, **metadata: Any) -> str:
    """
    Store *data* in GridFS and return the new file’s ObjectId as **str**.
    Extra kwargs land in the `metadata` field (e.g. source=<original_id>).
    """
    return str(_fs.put(data,
                       filename=filename,
                       content_type=content_type,
                       metadata=metadata))

def get_bytes(file_id: str) -> bytes:
    """Fetch raw bytes for *file_id* from GridFS."""
    return _fs.get(ObjectId(file_id)).read()

def get_gridout(file_id: str):
    """Return the underlying GridOut (useful for Flask `Response`)."""
    return _fs.get(ObjectId(file_id))

__all__ = ["put_file", "get_bytes", "get_gridout"]