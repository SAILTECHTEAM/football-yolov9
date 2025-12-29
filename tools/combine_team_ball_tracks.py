import os
import argparse

# Combine JSONL files of player and ball tracking
def combine_jsonl_files_streaming(player_jsonl_path, ball_jsonl_path, output_path):
    """
    Combines two JSONL files by appending ball_jsonl after player_jsonl in a streaming fashion.
    More memory efficient for large files.
    """
    count1 = 0
    count2 = 0
    # Create the directory for output if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as out_file:
        # Copy contents from player_jsonl
        with open(player_jsonl_path, 'r') as f1:
            for line in f1:
                if line.strip():  # Skip empty lines
                    out_file.write(line)
                    count1 += 1
        
        # Append contents from ball_jsonl
        with open(ball_jsonl_path, 'r') as f2:
            for line in f2:
                if line.strip():  # Skip empty lines
                    out_file.write(line)
                    count2 += 1
    
    print(f"Successfully combined files:")
    print(f"- {player_jsonl_path}: {count1} records")
    print(f"- {ball_jsonl_path}: {count2} records")
    print(f"- {output_path}: {count1 + count2} records total")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Combine two JSONL files into one.")
    parser.add_argument("--player-jsonl", type=str, required=True, help="Path to the first JSONL file (e.g., player tracks).")
    parser.add_argument("--ball-jsonl", type=str, required=True, help="Path to the second JSONL file (e.g., ball tracks).")
    parser.add_argument("--output-jsonl", type=str, required=True, help="Path to the output combined JSONL file.")

    args = parser.parse_args()
    
    combine_jsonl_files_streaming(args.player_jsonl, args.ball_jsonl, args.output_jsonl)