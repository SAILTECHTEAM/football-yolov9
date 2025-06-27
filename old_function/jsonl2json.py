import json
import argparse
from pathlib import Path

def convert_jsonl_to_json(jsonl_path, json_path=None):
    jsonl_path = Path(jsonl_path)

    if json_path is None:
        json_path = jsonl_path.with_suffix(".json")
    else:
        json_path = Path(json_path)

    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # skip empty lines
                data.append(json.loads(line))

    with open(json_path, "w", encoding="utf-8") as out_f:
        json.dump(data, out_f, ensure_ascii=False, indent=4)

    print(f"✅ Converted {jsonl_path} → {json_path} with {len(data)} records.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL to JSON array format.")
    parser.add_argument("jsonl_path", help="Path to input .jsonl file")
    parser.add_argument("--out", help="Optional output .json path (default: same name with .json extension)")
    args = parser.parse_args()

    convert_jsonl_to_json(args.jsonl_path, args.out)
