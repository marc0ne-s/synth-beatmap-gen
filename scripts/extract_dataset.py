import os
import json
import math
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# --- Constants from feasibility_checker.py ---
DIFF_MAP = {"Easy": 0, "Normal": 1, "Hard": 2, "Expert": 3, "Master": 4}

def compute_3d_distance(n1, n2):
    return math.sqrt((n2["x"] - n1["x"])**2 + (n2["y"] - n1["y"])**2 + (n2.get("z", 0) - n1.get("z", 0))**2)

def angle_between_vectors(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 == 0 or mag2 == 0: return 0.0
    cos = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos))

# --- Extraction Logic ---

def extract_features_from_map(gold_path, report_path):
    try:
        with open(gold_path, 'r') as f:
            gold = json.load(f)
        with open(report_path, 'r') as f:
            report = json.load(f)
    except Exception:
        return None

    extracted = []
    for diff_name, notes in gold.get("difficulties", {}).items():
        if diff_name not in DIFF_MAP: continue
        
        diff_report = report.get("difficulties", {}).get(diff_name, {})
        if not diff_report: continue
        
        label = 1 if diff_report.get("overall_pass", False) else 0
        
        # note-level features
        # split by hand
        hands = {0: [], 1: []}
        for n in notes:
            hands[n.get("type", 0)].append(n)
            
        note_features = []
        for h_type, h_notes in hands.items():
            for i in range(len(h_notes)):
                n = h_notes[i]
                dt = 0
                dist = 0
                vel = 0
                ang = 0
                
                if i > 0:
                    prev = h_notes[i-1]
                    dt = n["time"] - prev["time"]
                    dist = compute_3d_distance(prev, n)
                    if dt > 0:
                        vel = dist / (dt / 1000.0)
                        
                if i > 1:
                    prev1 = h_notes[i-1]
                    prev2 = h_notes[i-2]
                    v1 = (prev1["x"] - prev2["x"], prev1["y"] - prev2["y"])
                    v2 = (n["x"] - prev1["x"], n["y"] - prev1["y"])
                    ang = angle_between_vectors(v1, v2)
                
                # normalized x, y (play area is -4 to 4, -3 to 3)
                nx = n["x"] / 4.0
                ny = n["y"] / 3.0
                
                note_features.append({
                    "hand": h_type,
                    "x": nx,
                    "y": ny,
                    "dt": min(dt / 1000.0, 5.0), # cap dt at 5s
                    "dist": dist,
                    "vel": min(vel, 20.0), # cap velocity
                    "ang": ang / 180.0, # normalize angle
                    "time": n["time"] # for sorting later
                })
        
        # sort by time
        note_features.sort(key=lambda x: x["time"])
        
        # global features
        total_notes = len(notes)
        if total_notes == 0: continue
        
        balance = diff_report.get("balance", {})
        l_pct = balance.get("left_pct", 0.5)
        r_pct = balance.get("right_pct", 0.5)
        imb = balance.get("imbalance_score", 0.0)
        
        duration = diff_report.get("duration_s", 0.0)
        avg_nps = diff_report.get("avg_nps", 0.0)
        ramp = diff_report.get("density_ramp", {}).get("ramp_ratio", 1.0)
        
        map_features = [
            DIFF_MAP[diff_name] / 4.0, # normalized difficulty
            total_notes / 2000.0, # normalized note count
            duration / 300.0, # normalized duration
            avg_nps / 20.0, # normalized NPS
            l_pct,
            r_pct,
            imb,
            ramp / 5.0 # normalized ramp
        ]
        
        extracted.append({
            "difficulty": diff_name,
            "seq": [[f["hand"], f["x"], f["y"], f["dt"], f["dist"], f["vel"], f["ang"]] for f in note_features],
            "global": map_features,
            "label": label,
            "uuid": gold.get("uuid")
        })
        
    return extracted

def process_batch(file_pairs):
    results = []
    for g, r in file_pairs:
        res = extract_features_from_map(g, r)
        if res:
            results.extend(res)
    return results

if __name__ == "__main__":
    base_path = "/Volumes/Second-Brain-1/AI/Synth/"
    gold_dir = Path(base_path) / "evaluation/phase12b/gold_standard/"
    report_dir = Path(base_path) / "evaluation/phase12b/feasibility_reports/"
    
    gold_files = sorted(list(gold_dir.glob("*.json")))
    print(f"Found {len(gold_files)} gold bundles.")
    
    file_pairs = []
    for g in gold_files:
        r = report_dir / g.name
        if r.exists():
            file_pairs.append((str(g), str(r)))
            
    print(f"Matched {len(file_pairs)} pairs.")
    
    start_time = time.time()
    num_workers = 12
    chunk_size = len(file_pairs) // (num_workers * 2)
    chunks = [file_pairs[i:i + chunk_size] for i in range(0, len(file_pairs), chunk_size)]
    
    all_data = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for batch_res in executor.map(process_batch, chunks):
            all_data.extend(batch_res)
            
    print(f"Extracted {len(all_data)} difficulty-map samples.")
    print(f"Time taken: {time.time() - start_time:.2f}s")
    
    output_path = "/Users/marcus/.gemini/antigravity/scratch/dataset_v0.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(all_data, f)
    print(f"Saved dataset to {output_path}")
