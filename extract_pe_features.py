import lief
import json
import math
import sys
import argparse
import os
from collections import Counter

# ---------------------------
# Helper Functions
# ---------------------------

def to_primitive(x):
    if x is None:
        return None
    if isinstance(x, (int, float, str, bool)):
        return x
    try:
        if hasattr(x, "__iter__") and not isinstance(x, (str, bytes, dict)):
            return [to_primitive(v) for v in x]
    except:
        pass
    try:
        return int(x)
    except:
        try:
            return str(x)
        except:
            return None

def byte_histogram(data):
    hist = [0] * 256
    for b in data:
        hist[b] += 1
    total = len(data)
    if total > 0:
        return [h / total for h in hist]
    return hist

def byteentropy(data):
    window_size = 2048
    entropies = []
    for i in range(0, len(data), window_size):
        chunk = data[i:i + window_size]
        if not chunk:
            continue
        counter = Counter(chunk)
        entropy = 0.0
        for c in counter.values():
            p = c / len(chunk)
            entropy -= p * math.log2(p)
        entropies.append(entropy)
    return entropies

def safe_get(obj, *attrs, default=None):
    for a in attrs:
        if hasattr(obj, a):
            try:
                return getattr(obj, a)
            except:
                continue
    return default


# ---------------------------
# MAIN FEATURE EXTRACTOR
# ---------------------------

def extract_features(path):

    pe = lief.parse(path)
    if pe is None:
        raise RuntimeError("Not a valid PE file (lief.parse returned None).")

    with open(path, "rb") as f:
        raw_bytes = f.read()

    optional = getattr(pe, "optional_header", None)
    header = getattr(pe, "header", None)

    entrypoint = safe_get(optional, "addressof_entrypoint", "entrypoint", default=None)
    timestamp = safe_get(header, "time_date_stamp", "time_date_stamps", "time_date", default=None)

    virtual_size = safe_get(pe, "virtual_size", "size", default=None)

    features = {
        "appeared": 0,
        "histogram": byte_histogram(raw_bytes),
        "byteentropy": byteentropy(raw_bytes),
        "strings": {"num_strings": 0},
        "general": {
            "size": to_primitive(virtual_size),
            "entrypoint": to_primitive(entrypoint),
            "timestamp": to_primitive(timestamp),
            "number_of_sections": len(getattr(pe, "sections", []))
        },
        "header": {
            "machine": to_primitive(safe_get(header, "machine", default=None)),
            "characteristics": to_primitive(safe_get(header, "characteristics", default=None)),
            "subsystem": to_primitive(safe_get(optional, "subsystem", default=None)),
        },
        "section": {},
        "imports": {},
        "exports": {}
    }

    # Sections
    for sec in getattr(pe, "sections", []):
        name = sec.name if sec.name else f"sec_{len(features['section'])}"
        features["section"][name] = {
            "virtual_size": to_primitive(sec.virtual_size),
            "size": to_primitive(sec.size),
            "entropy": to_primitive(sec.entropy)
        }

    # Imports
    for entry in getattr(pe, "imports", []):
        libname = entry.name
        funcs = []
        for e in entry.entries:
            funcs.append(to_primitive(safe_get(e, "name", "value", default=None)))
        features["imports"][libname] = funcs

    # Exports
    expf = []
    try:
        for e in getattr(pe, "exported_functions", []):
            expf.append(to_primitive(e))
    except:
        pass
    features["exports"] = {"functions": expf}

    return features


# ---------------------------
# MAIN EXECUTION
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    # Folder chứa output
    OUTPUT_FOLDER = r"E:\data for life\model AI 1\xgboost-input_moi\json extract file"

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    basename = os.path.basename(args.file)
    name_no_ext = os.path.splitext(basename)[0]
    out_path = os.path.join(OUTPUT_FOLDER, name_no_ext + ".json")

    feat = extract_features(args.file)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(feat, f)

    print("Extracted ->", out_path)
