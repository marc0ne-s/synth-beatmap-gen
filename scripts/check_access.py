import os
import json

base_path = "/Volumes/Second-Brain-1/AI/Synth/"
gold_path = os.path.join(base_path, "evaluation/phase12b/gold_standard/")
reports_path = os.path.join(base_path, "evaluation/phase12b/feasibility_reports/")

print(f"Checking access to {gold_path}...")
try:
    files = os.listdir(gold_path)
    print(f"Found {len(files)} files.")
    
    # Try reading one
    if files:
        with open(os.path.join(gold_path, files[0]), 'r') as f:
            data = json.load(f)
            print("Successfully read one gold bundle.")
except Exception as e:
    print(f"Error: {e}")
