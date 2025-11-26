import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd 


from flask import Flask, jsonify, request
from flask_cors import CORS

# App setup
app = Flask(__name__)
CORS(app)

# Config
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VALID_POSITIONS = {"rb", "qb", "wr", "te"}

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------- Helpers ----------
def json_error(message: str, code: int = 400):
    """Return a JSON error response."""
    return jsonify({"error": message}), code


def require_json():
    """Get JSON payload safely (returns empty dict if none)."""
    return request.get_json(silent=True) or {}


def validate_position(position_raw: Optional[str]) -> Optional[str]:
    """Normalize and validate position string. Returns lowercased position or None."""
    if not position_raw:
        return None
    pos = str(position_raw).strip().lower()
    return pos if pos in VALID_POSITIONS else None


def read_text_file(path: Path) -> str:
    """Read a text file and return contents (raises)."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------- Routes ----------
@app.route("/fetch_data", methods=["POST", "OPTIONS"], endpoint = "fetch_data_endpoint")
def fetch_data():
    """
    Expects JSON:
      - position: rb|qb|wr|te (required)
      - type: predictions (default)
      - model: xgb (default)
      - limit: int (default 50, capped 1..1000)
      - offset: int (default 0, min 0)
    Returns paginated parquet data as records with total/offset/limit/has_more.
    """

    data = require_json()
    position = validate_position(data.get("position"))
    if not position:
        return json_error("Invalid or missing position", 400)

    df_type = str(data.get("type", "predictions"))
    model = str(data.get("model", "xgb"))

    # parse numeric params with safe fallbacks
    try:
        limit = int(data.get("limit", 50))
    except Exception:
        limit = 50
    try:
        offset = int(data.get("offset", 0))
    except Exception:
        offset = 0

    # bounds
    if limit < 1 or limit > 1000:
        limit = 50
    if offset < 0:
        offset = 0

    parquet_path = DATA_DIR / f"{position}_{df_type}_{model}.parquet"
    if not parquet_path.exists():
        return json_error(f"Data file not found: {parquet_path}", 404)

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        logger.exception("Failed to read parquet file")
        return json_error(f"Error reading parquet file: {str(e)}", 500)

    total_rows = len(df)
    paginated_df = df.iloc[offset: offset + limit]
    records = paginated_df.to_dict(orient="records")

    return jsonify({
        "data": records,
        "total": total_rows,
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < total_rows
    })


@app.route("/get_error", methods=["POST", "OPTIONS"], endpoint = "get_error_endpoint")
def get_error():
    """
    Expects JSON:
      - position: rb|qb|wr|te (required)
    Returns the text contents of {position}_error.txt as error_text.
    """

    data = require_json()
    position = validate_position(data.get("position"))
    if not position:
        return json_error("Invalid or missing position", 400)

    error_path = DATA_DIR / f"{position}_error.txt"
    if not error_path.exists():
        return json_error(f"Error file not found: {error_path}", 404)

    try:
        content = read_text_file(error_path)
        return jsonify({"error_text": content})
    except Exception as e:
        logger.exception("Failed to read error file")
        return json_error(f"Error reading error file: {str(e)}", 500)


@app.route("/get_features", methods=["POST", "OPTIONS"], endpoint = "get_features_endpoint")
def get_features():
    """
    Expects JSON:
      - position: rb|qb|wr|te (required)
    Returns a list of non-empty lines from {position}_features.txt.
    """

    data = require_json()
    position = validate_position(data.get("position"))
    if not position:
        return json_error("Position is required and must be one of rb/qb/wr/te", 400)

    features_path = DATA_DIR / f"{position}_features.txt"
    if not features_path.exists():
        return json_error(f"Features file not found: {features_path}", 404)

    try:
        with open(features_path, "r", encoding="utf-8") as f:
            features = [line.strip() for line in f.readlines() if line.strip()]
        return jsonify(features)
    except Exception as e:
        logger.exception("Failed to read features file")
        return json_error(f"Error reading features file: {str(e)}", 500)


@app.route("/get_perm_importance", methods=["POST", "OPTIONS"], endpoint = "get_perm_importance_endpoint")
def get_perm_importance():
    """
    Expects JSON:
      - position: rb|qb|wr|te (required)
    Reads {position}_perm_importance.txt and extracts top features with importance and std.
    Returns: {"features": [{"feature": name, "importance": float, "std": float}, ...]}
    """

    data = require_json()
    position = validate_position(data.get("position"))
    if not position:
        return json_error("Invalid or missing position", 400)

    perm_path = DATA_DIR / f"{position}_perm_importance.txt"
    if not perm_path.exists():
        return json_error(f"File not found: {perm_path}", 404)

    try:
        content = read_text_file(perm_path).strip()

        # pattern matches "12.34 +/- 0.56" (with optional decimals)
        pattern = r'(\d+\.?\d*)\s*\+\/-\s*(\d+\.?\d*)'
        matches = list(re.finditer(pattern, content))

        features_list: List[Dict[str, Any]] = []
        for i, match in enumerate(matches):
            importance = float(match.group(1))
            std = float(match.group(2))
            prev_end = matches[i - 1].end() if i > 0 else 0
            # text between previous match end and current match start should contain the feature name
            text_before = content[prev_end:match.start()].strip()
            # attempt to extract a plausible feature token at the end of text_before
            feature_name = None
            fm = re.search(r'([A-Za-z0-9_/.-]+)\s*$', text_before)
            if fm:
                feature_name = fm.group(1)
            else:
                # fallback: strip trailing numbers
                feature_name = re.sub(r'\s+\d+\.?\d*\s*$', '', text_before).strip()
            if feature_name:
                features_list.append({
                    "feature": feature_name,
                    "importance": importance,
                    "std": std
                })

        # sort and take top 15 (or fewer)
        features_list.sort(key=lambda x: x["importance"], reverse=True)
        top_features = features_list[:15]

        return jsonify({"features": top_features})
    except Exception as e:
        logger.exception("Failed to parse perm importance file")
        return json_error(f"Error reading file: {str(e)}", 500)


# Root health-check
@app.route("/", methods=["GET"], endpoint = "root_endpoint")
def root():
    return jsonify({"status": "ok", "endpoints": ["/fetch_data", "/get_error", "/get_features", "/get_perm_importance"]})