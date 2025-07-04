import ijson
import json

def convert_large_json_to_jsonl(json_path, jsonl_path):
    """
    Converts a large JSON array file to JSONL format without loading all data into memory.

    Args:
        json_path (str): Path to the input .json file.
        jsonl_path (str): Path to the output .jsonl file.
    """
    with open(json_path, 'r') as f_in, open(jsonl_path, 'w') as f_out:
        # Assumes the root element is a list: [ {...}, {...}, ... ]
        parser = ijson.items(f_in, 'item')
        for obj in parser:
            f_out.write(json.dumps(obj, default=float) + '\n')

    print(f"✅ Converted to {jsonl_path}")

# Example usage:
if __name__ == "__main__":
    convert_large_json_to_jsonl("./runs/detect/test_4k-2h/team_tracking.json", "./runs/detect/test_4k-2h/team_tracking.jsonl")