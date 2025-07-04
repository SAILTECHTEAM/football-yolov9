import json

def get_max_frame_id(json_path):
    max_frame = -1

    with open(json_path, "r") as f:
        first_line = f.readline()
        if first_line.strip().startswith("{"):
            # JSONL format
            f.seek(0)  # reset file pointer
            for line in f:
                obj = json.loads(line.strip())
                frames = obj.get("frame_id", [])
                if frames:
                    max_frame = max(max_frame, max(frames))
        else:
            # Regular JSON array format
            f.seek(0)
            data = json.load(f)
            for obj in data:
                frames = obj.get("frame_id", [])
                if frames:
                    max_frame = max(max_frame, max(frames))

    print(f"📦 Largest frame_id found: {max_frame}")
    return max_frame

# Example usage
if __name__ == "__main__":
    json_path = "./runs/detect/test_4k-2h/team_tracking.jsonl"  # or .json
    max = get_max_frame_id(json_path)