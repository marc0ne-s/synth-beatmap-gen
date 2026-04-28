#!/usr/bin/env python3
import os
import json
import numpy as np
from pathlib import Path

def validate_map(map_path):
    print(f"[?] Validating: {map_path.name}")
    with open(map_path, "r") as f:
        data = json.load(f)
    
    results = []
    for diff, notes in data["difficulties"].items():
        if not notes:
            results.append(f"  [!] {diff}: EMPTY MAP")
            continue
            
        times = [n["time"] for n in notes]
        
        # 1. Coordinate check
        oob_count = 0
        for n in notes:
            if abs(n["x"]) > 8.0 or abs(n["y"]) > 6.0: # SRT typical range
                oob_count += 1
        
        # 2. Overlap/Gap check
        # Notes of same type should have min 40ms gap
        overlap_count = 0
        for h in [0, 1]:
            h_notes = [n for n in notes if n["type"] == h]
            h_times = sorted([n["time"] for n in h_notes])
            for i in range(1, len(h_times)):
                if h_times[i] - h_times[i-1] < 40.0:
                    overlap_count += 1
                    
        # 3. Density check
        # Heuristic: Warn if < 0.5 notes per second across the track
        duration_sec = (max(times) - min(times)) / 1000.0 if times else 1
        nps = len(notes) / duration_sec
        
        status = f"  {diff}: {len(notes)} notes | {nps:.1f} NPS"
        if oob_count > 0: status += f" | ⚠️ {oob_count} OOB"
        if overlap_count > 0: status += f" | ❌ {overlap_count} OVERLAPS"
        if nps < 0.8: status += " | ⚠️ LOW DENSITY"
        
        results.append(status)
    
    return "\n".join(results)

def main():
    root = Path("/Volumes/Second-Brain-1/AI/Synth/evaluation/phase12b/gold_standard")
    maps = list(root.glob("*.json"))
    print(f"[+] Found {len(maps)} maps to validate.")
    
    total_maps = len(maps)
    oob_maps = 0
    overlap_maps = 0
    low_density_maps = 0
    empty_maps = 0
    
    for m in maps:
        with open(m, "r") as f:
            data = json.load(f)
        
        for diff, notes in data["difficulties"].items():
            if not notes:
                empty_maps += 1
                continue
            
            times = [n["time"] for n in notes]
            
            # 1. Coordinate check
            for n in notes:
                if abs(n["x"]) > 12.0 or abs(n["y"]) > 10.0: # Broad sanity
                    oob_maps += 1
                    break
            
            # 2. Overlap/Gap check
            found_overlap = False
            for h in [0, 1]:
                h_notes = [n for n in notes if n["type"] == h]
                h_times = sorted([n["time"] for n in h_notes])
                for i in range(1, len(h_times)):
                    if h_times[i] - h_times[i-1] < 40.0:
                        overlap_maps += 1
                        found_overlap = True
                        break
                if found_overlap: break
                    
            # 3. Density check
            times = [n["time"] for n in notes]
            duration_sec = (max(times) - min(times)) / 1000.0 if len(times) > 1 else 1.0
            if duration_sec == 0: duration_sec = 1.0
            nps = len(notes) / duration_sec
            if nps < 0.5:
                low_density_maps += 1

    print("\n" + "="*50)
    print("FINAL CORPUS VALIDATION REPORT")
    print("="*50)
    print(f"Total Maps Validated: {total_maps}")
    print(f"Empty Maps:          {empty_maps}")
    print(f"Coordinate OOB:       {oob_maps}")
    print(f"Temporal Overlaps:    {overlap_maps}")
    print(f"Low Density Warnings: {low_density_maps}")
    print("="*50)

if __name__ == "__main__":
    main()
