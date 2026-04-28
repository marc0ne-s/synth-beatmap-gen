#!/usr/bin/env python3
"""
parse_beatmap_data.py

Parse SynthRiders `beatmap.meta.bin` JSON files into structured note arrays
for machine learning. Extracts metadata, note counts, and per-difficulty note lists.

Usage:
    python parse_beatmap_data.py <beatmap.meta.bin> [--output <json>]
    python parse_beatmap_data.py --batch <extracted_dir> [--output-dir <out_dir>]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional


def parse_beatmap(bin_path: str) -> Optional[dict]:
    """
    Parse a single beatmap.meta.bin file.
    Returns structured dict or None on failure.
    """
    try:
        with open(bin_path, "r", encoding="utf-8-sig", errors="replace") as f:
            raw = f.read()
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as e:
        return {"status": "failed", "error": str(e), "path": bin_path}

    result = {
        "status": "success",
        "path": bin_path,
        "metadata": {},
        "difficulties": {},
        "stats": {},
    }

    # Extract metadata (strip heavy base64 fields)
    result["metadata"] = {
        "name": data.get("Name", ""),
        "author": data.get("Author", ""),
        "bpm": data.get("BPM", 0.0),
        "offset": data.get("Offset", 0.0),
        "beatmapper": data.get("Beatmapper", ""),
        "editor_version": data.get("EditorVersion", ""),
        "production_mode": data.get("ProductionMode", False),
        "modified_time": data.get("ModifiedTime", 0),
    }

    # Parse each difficulty tier
    total_notes = 0
    track_data = data.get("Track", {})
    effects_data = data.get("Effects", {})
    jumps_data = data.get("Jumps", {})
    crouchs_data = data.get("Crouchs", {})
    slides_data = data.get("Slides", {})

    for diff in ["Easy", "Normal", "Hard", "Expert", "Master", "Custom"]:
        notes_dict = track_data.get(diff, {})
        effects_dict = effects_data.get(diff, {})
        jumps_dict = jumps_data.get(diff, {})
        crouchs_dict = crouchs_data.get(diff, {})
        slides_dict = slides_data.get(diff, {})

        if not notes_dict:
            continue

        # Flatten notes into a sorted list
        note_list = []
        for time_key, notes_at_time in sorted(notes_dict.items(), key=lambda x: float(x[0])):
            timestamp = float(time_key)
            for note in notes_at_time:
                pos = note.get("Position", [0.0, 0.0, 0.0])
                note_list.append({
                    "time": timestamp,
                    "x": pos[0] if len(pos) > 0 else 0.0,
                    "y": pos[1] if len(pos) > 1 else 0.0,
                    "z": pos[2] if len(pos) > 2 else 0.0,
                    "type": note.get("Type", 0),
                    "direction": note.get("Direction", 0),
                    "combo_id": note.get("ComboId", -1),
                    "id": note.get("Id", ""),
                    "segments": note.get("Segments"),
                })

        result["difficulties"][diff] = {
            "note_count": len(note_list),
            "notes": note_list,
            "effect_count": len(effects_dict) if isinstance(effects_dict, dict) else 0,
            "jump_count": len(jumps_dict) if isinstance(jumps_dict, dict) else 0,
            "crouch_count": len(crouchs_dict) if isinstance(crouchs_dict, dict) else 0,
            "slide_count": len(slides_dict) if isinstance(slides_dict, dict) else 0,
        }
        total_notes += len(note_list)

    result["stats"] = {
        "total_notes": total_notes,
        "difficulty_count": len(result["difficulties"]),
        "has_base64_artwork": bool(data.get("ArtworkBytes")),
        "file_size_kb": os.path.getsize(bin_path) / 1024,
    }

    return result


def batch_parse(extracted_dir: str, output_dir: str) -> None:
    """Parse all beatmap.meta.bin files in extracted subdirectories."""
    os.makedirs(output_dir, exist_ok=True)

    entries = []
    for entry in os.listdir(extracted_dir):
        bin_path = os.path.join(extracted_dir, entry, "beatmap.meta.bin")
        if os.path.exists(bin_path):
            entries.append(bin_path)

    total = len(entries)
    print(f"[+] Found {total} beatmap.meta.bin files to parse")

    results = []
    failed = []
    for i, bin_path in enumerate(entries, 1):
        result = parse_beatmap(bin_path)
        if result["status"] == "success":
            results.append(result)
        else:
            failed.append(result)

        if i % 500 == 0 or i == total:
            print(f"[{i}/{total}] OK={len(results)} FAIL={len(failed)}")

    # Write individual parsed files (strip full note arrays for index)
    index = []
    for r in results:
        # Save full parsed data as JSON
        uuid = os.path.basename(os.path.dirname(r["path"]))
        out_path = os.path.join(output_dir, f"{uuid}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)

        # Index entry (metadata + stats only, no notes to keep it small)
        index.append({
            "uuid": uuid,
            "metadata": r["metadata"],
            "stats": r["stats"],
            "difficulties": {
                k: {sk: sv for sk, sv in v.items() if sk != "notes"}
                for k, v in r["difficulties"].items()
            },
        })

    # Write master index
    index_path = os.path.join(output_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_maps": len(results),
            "failed": len(failed),
            "maps": index,
        }, f, indent=2, ensure_ascii=False)

    print(f"[+] Saved {len(results)} parsed files to {output_dir}")
    print(f"[+] Saved master index to {index_path}")

    if failed:
        failed_path = os.path.join(output_dir, "parse_failed.log")
        with open(failed_path, "w") as f:
            for item in failed:
                f.write(f"{item['path']} | {item['error']}\n")
        print(f"[+] {len(failed)} failures logged to {failed_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse SynthRiders beatmap.meta.bin files")
    parser.add_argument("file", nargs="?", help="Single beatmap.meta.bin to parse")
    parser.add_argument("--batch", help="Directory with extracted beatmap folders")
    parser.add_argument("--output", "-o", help="Output JSON file for single parse")
    parser.add_argument("--output-dir", default="dataset/parsed", help="Output dir for batch parse")
    args = parser.parse_args()

    if args.batch:
        base = "/Volumes/Second-Brain-1/AI/Synth"
        extracted = os.path.join(base, args.batch) if not os.path.isabs(args.batch) else args.batch
        out_dir = os.path.join(base, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
        batch_parse(extracted, out_dir)
    elif args.file:
        result = parse_beatmap(args.file)
        json_str = json.dumps(result, indent=2, ensure_ascii=False)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"Wrote to {args.output}")
        else:
            print(json_str)
    else:
        parser.error("Provide either --batch or a file path")


if __name__ == "__main__":
    main()
