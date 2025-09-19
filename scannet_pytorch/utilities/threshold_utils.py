import json

def load_threshold(json_path, key="threshold"):
    """
    Load threshold from a sweep JSON file (ops_val.examples.thresholds.json).
    Default key is "threshold" = the chosen one per --prefer.
    """
    with open(json_path, "r") as f:
        obj = json.load(f)
    return float(obj.get(key, 0.5))

