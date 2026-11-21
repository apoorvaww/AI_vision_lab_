"""Centralized configuration for AI VisionLab."""

import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)


def positive_int(name, value):
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def bounded_int(name, value, minimum, maximum):
    result = positive_int(name, value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return result


HOST = os.environ.get("HOST", "0.0.0.0")
PORT = bounded_int("PORT", os.environ.get("PORT", "5000"), 1, 65535)
ENV = os.environ.get("FLASK_ENV", "development").strip().lower()
DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1", "yes")

configured_secret = (
    os.environ.get("STAGE1_SECRET_KEY")
    or os.environ.get("SECRET_KEY")
)

placeholders = {
    "dev-secret-key-change-me",
    "change-this-to-a-secure-random-key-in-production",
}

if ENV == "production":
    if not configured_secret or configured_secret in placeholders:
        raise RuntimeError(
            "Set a secure STAGE1_SECRET_KEY. Generate one with: openssl rand -hex 32"
        )
    if len(configured_secret) < 32:
        raise RuntimeError("Production secret must contain at least 32 characters.")
    SECRET_KEY = configured_secret
else:
    SECRET_KEY = configured_secret or secrets.token_hex(32)

upload_limit = os.environ.get("MAX_UPLOAD_MB")
if upload_limit is None:
    upload_limit = os.environ.get("STAGE1_MAX_UPLOAD_MB", "2048")

MAX_UPLOAD_MB = positive_int("MAX_UPLOAD_MB", upload_limit)
MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024

MAX_EXTRACT_FILES = positive_int(
    "MAX_EXTRACT_FILES", os.environ.get("MAX_EXTRACT_FILES", "10000")
)
MAX_EXTRACT_BYTES = positive_int(
    "MAX_EXTRACT_BYTES",
    os.environ.get("MAX_EXTRACT_BYTES", str(1024 * 1024 * 1024)),
)
MAX_EXTRACT_RATIO = positive_int(
    "MAX_EXTRACT_RATIO", os.environ.get("MAX_EXTRACT_RATIO", "100")
)

data_path = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "webapp_data"))).expanduser()
if not data_path.is_absolute():
    data_path = BASE_DIR / data_path

DATA_DIR = str(data_path.resolve())
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
RESULTS_DIR = os.path.join(DATA_DIR, "results")


def init_app_dirs():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


init_app_dirs()
