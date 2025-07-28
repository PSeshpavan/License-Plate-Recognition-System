import os
from typing import Any

from bson import ObjectId
from pymongo import MongoClient
import gridfs
import dotenv
dotenv.load_dotenv()



# ── Mongo connection settings ──────────────────────────────────────────────
MONGO_URI: str = os.getenv("MONGO_URI")
DATABASE_NAME: str   = os.getenv("DATABASE_NAME")

_client = MongoClient(MONGO_URI)
_db     = _client[DATABASE_NAME]
_fs     = gridfs.GridFS(_db)

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