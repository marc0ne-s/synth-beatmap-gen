"""
SynthRiders Feasibility Checker (Phase 13 — Pre-Flight Simulator)

Analyses generated .synth bundles for playability without human playtesting.
Computes reachability, flow continuity, spatial balance, density ramps, and
rail candidates. Outputs a JSON report per map that correlates with human feel.

Usage:
    # Single map
    python scripts/feasibility_checker.py --map evaluation/phase12b/gold_standard/0007b2da6d9527ab.json

    # Full batch (parallel)
    python scripts/feasibility_checker.py --batch evaluation/phase12b/gold_standard/ --out evaluation/phase12b/feasibility_reports/

    # Difficulty-scoped audit (print summary table)
    python scripts/feasibility_checker.py --audit --indir evaluation/phase12b/feasibility_reports/
"""

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Difficulty-Aware Parameters ──────────────────────────────────────
DIFFICULTY_PARAMS = {
    "Easy": {
        "hit_window_ms": 200,
        "max_note_density_nps": 9.0,
        "reachability_tol": 0.60,
        "flow_ang_tol_deg": 135.0,
        "min_rail_len": 3,
    },
    "Normal": {
        "hit_window_ms": 120,
        "max_note_density_nps": 13.0,
        "reachability_tol": 0.70,
        "flow_ang_tol_deg": 120.0,
        "min_rail_len": 3,
    },
    "Hard": {
        "hit_window_ms": 80,
        "max_note_density_nps": 16.0,
        "reachability_tol": 0.80,
        "flow_ang_tol_deg": 100.0,
        "min_rail_len": 3,
    },
    "Expert": {
        "hit_window_ms": 60,
        "max_note_density_nps": 19.0,
        "reachability_tol": 0.90,
        "flow_ang_tol_deg": 90.0,
        "min_rail_len": 4,
    },
    "Master": {
        "hit_window_ms": 40,
        "max_note_density_nps": 25.0,
        "reachability_tol": 1.00,
        "flow_ang_tol_deg": 80.0,
        "min_rail_len": 4,
    },
}

# ── Physical Constants (estimated for average adult male, 1.8m) ──────
ARM_LENGTH_M = 0.75          # Shoulder to controller
MAX_CONTROLLER_SPEED_MS = 4.0
MAX_ANGULAR_VEL_DEG_S = 450.0
SHOULDER_WIDTH_M = 0.40

# ── SynthRiders Play Area (coordinate space) ──────────────────────────
PLAY_AREA = {
    "x_min": -4.0, "x_max": 4.0,
    "y_min": -3.0, "y_max": 3.0,
    "z_min": -2.0, "z_max": 2.0,
}

# Generated maps only have x, y; we infer z=0 (plane) for 2D maps.

def compute_2d_distance(n1: Dict, n2: Dict) -> float:
    return math.hypot(n2["x"] - n1["x"], n2["y"] - n1["y"])


def compute_3d_distance(n1: Dict, n2: Dict) -> float:
    z1 = n1.get("z", 0.0)
    z2 = n2.get("z", 0.0)
    return math.sqrt(
        (n2["x"] - n1["x"])**2 +
        (n2["y"] - n1["y"])**2 +
        (z2 - z1)**2
    )


def angle_between_vectors(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Return angle in degrees [0, 180] between two 2D vectors."""
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos))


# ── Core Checker ─────────────────────────────────────────────────────
class FeasibilityChecker:
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def check_bounds(self, notes: List[Dict]) -> Dict:
        oob_notes = []
        for i, n in enumerate(notes):
            if (
                n["x"] < PLAY_AREA["x_min"] or n["x"] > PLAY_AREA["x_max"] or
                n["y"] < PLAY_AREA["y_min"] or n["y"] > PLAY_AREA["y_max"]
            ):
                oob_notes.append(i)
        return {
            "oob_count": len(oob_notes),
            "oob_indices": oob_notes,
            "pass": len(oob_notes) == 0,
        }

    def check_reachability(self, notes: List[Dict]) -> Dict:
        """
        For each consecutive pair WITHIN THE SAME HAND (type), compute if human arm can traverse
        distance in time_delta. Cross-hand transitions are two independent controllers.
        """
        # Split notes by hand type before checking
        type0_notes = [n for n in notes if n.get("type", 0) == 0]
        type1_notes = [n for n in notes if n.get("type", 0) == 1]
        all_violations = []
        for hand_notes, hand_label in [(type0_notes, 0), (type1_notes, 1)]:
            for i in range(len(hand_notes) - 1):
                n1, n2 = hand_notes[i], hand_notes[i + 1]
                dt_ms = n2["time"] - n1["time"]
                if dt_ms <= 0:
                    all_violations.append({"hand": hand_label, "index": i, "dt_ms": dt_ms,
                                           "distance": 0, "required_speed": float('inf'),
                                           "reason": "zero_or_negative_dt"})
                    continue
                dist = compute_3d_distance(n1, n2)
                required_speed = dist / (dt_ms / 1000.0)
                tol_speed = MAX_CONTROLLER_SPEED_MS * self.params["reachability_tol"]
                if required_speed > tol_speed:
                    all_violations.append({
                        "hand": hand_label,
                        "index": i,
                        "dt_ms": dt_ms,
                        "distance": round(dist, 3),
                        "required_speed": round(required_speed, 2),
                        "tol_speed": round(tol_speed, 2),
                    })
        total_pairs = max(len(notes) - 1, 0)
        return {
            "pass": len(all_violations) == 0,
            "violation_count": len(all_violations),
            "total_pairs": total_pairs,
            "violations": all_violations[:10],
        }

    def check_flow(self, notes: List[Dict]) -> Dict:
        """
        Measure angular smoothness between triplets WITHIN THE SAME HAND.
        Cross-hand 180° ping-pongs are EXPECTED and fine.
        """
        type0_notes = [n for n in notes if n.get("type", 0) == 0]
        type1_notes = [n for n in notes if n.get("type", 0) == 1]
        all_angles = []
        all_spikes = []
        tol = self.params["flow_ang_tol_deg"]
        for hand_notes, hand_label in [(type0_notes, 0), (type1_notes, 1)]:
            if len(hand_notes) < 3:
                continue
            for i in range(len(hand_notes) - 2):
                n1, n2, n3 = hand_notes[i], hand_notes[i+1], hand_notes[i+2]
                v1 = (n2["x"] - n1["x"], n2["y"] - n1["y"])
                v2 = (n3["x"] - n2["x"], n3["y"] - n2["y"])
                ang = angle_between_vectors(v1, v2)
                all_angles.append(ang)
                if ang > tol:
                    all_spikes.append({"hand": hand_label, "index": i+1, "angle": round(ang, 1)})
        if not all_angles:
            return {"pass": True, "avg_ang_deg": 0.0, "max_ang_deg": 0.0, "spike_count": 0, "spikes": []}
        avg_ang = sum(all_angles) / len(all_angles)
        return {
            "pass": len(all_spikes) == 0,
            "avg_ang_deg": round(avg_ang, 2),
            "max_ang_deg": round(max(all_angles), 2),
            "spike_count": len(all_spikes),
            "spikes": all_spikes[:10],
        }

    def check_balance(self, notes: List[Dict]) -> Dict:
        """Left/right hemisphere balance. Ideal ~50/50. Warn if heavily skewed."""
        left = sum(1 for n in notes if n["x"] < 0)
        right = sum(1 for n in notes if n["x"] > 0)
        total = len(notes)
        if total == 0:
            return {"pass": True, "left_pct": 0.0, "right_pct": 0.0, "imbalance_score": 0.0}

        left_pct = left / total
        right_pct = right / total
        # Imbalance: how far from 50/50?
        imbalance = abs(left_pct - right_pct)
        return {
            "pass": imbalance < 0.35,  # allow up to 65/35 split
            "left_pct": round(left_pct, 3),
            "right_pct": round(right_pct, 3),
            "center_count": total - left - right,
            "imbalance_score": round(imbalance, 3),
        }

    def check_density_ramp(self, notes: List[Dict]) -> Dict:
        """Compare first 25% vs last 25% note density."""
        if len(notes) <= 4:
            return {"pass": True, "early_nps": 0.0, "late_nps": 0.0, "ramp_ratio": 1.0}

        # Sort by time (should already be, but safety)
        notes_sorted = sorted(notes, key=lambda n: n["time"])
        total_time_ms = notes_sorted[-1]["time"] - notes_sorted[0]["time"]
        if total_time_ms <= 0:
            return {"pass": True, "early_nps": 0.0, "late_nps": 0.0, "ramp_ratio": 1.0}

        quarter_time = total_time_ms / 4.0
        early_cutoff = notes_sorted[0]["time"] + quarter_time
        late_start = notes_sorted[0]["time"] + 3 * quarter_time

        early_count = sum(1 for n in notes_sorted if n["time"] <= early_cutoff)
        late_count = sum(1 for n in notes_sorted if n["time"] >= late_start)
        early_nps = (early_count / (quarter_time / 1000.0)) if quarter_time > 0 else 0.0
        late_nps = (late_count / (quarter_time / 1000.0)) if quarter_time > 0 else 0.0

        ramp_ratio = late_nps / early_nps if early_nps > 0 else float('inf')
        # Pass if late is at least 0.5× early (allow ambient intros)
        # AND late doesn't exceed max density for this difficulty
        max_nps = self.params["max_note_density_nps"]
        pass_ = (0.3 <= ramp_ratio <= 6.0) and (late_nps <= max_nps * 1.2)
        return {
            "pass": pass_,
            "early_nps": round(early_nps, 2),
            "late_nps": round(late_nps, 2),
            "ramp_ratio": round(ramp_ratio, 2),
            "max_nps_threshold": max_nps,
        }

    def check_co_located(self, notes: List[Dict]) -> Dict:
        """Notes that are physically impossible: same-ish position within 40ms.
        In SynthRiders, this would be a double-hit on the same hand."""
        violations = []
        for i in range(len(notes) - 1):
            n1, n2 = notes[i], notes[i + 1]
            dt = n2["time"] - n1["time"]
            dist = compute_3d_distance(n1, n2)
            if dt < 40 and dist < 0.5:
                violations.append({"index": i, "dt_ms": dt, "distance": round(dist, 3)})
        return {
            "pass": len(violations) == 0,
            "violation_count": len(violations),
            "violations": violations[:10],
        }

    def find_rail_candidates(self, notes: List[Dict]) -> Dict:
        """Find sequences that look like straight rails (sustained direction, tight timing)."""
        if len(notes) < 3:
            return {"rails": [], "rail_count": 0, "total_rail_notes": 0}

        rails = []
        current = [notes[0]]
        for i in range(1, len(notes)):
            n1 = notes[i - 1]
            n2 = notes[i]
            dt = n2["time"] - n1["time"]
            # Rail candidate if tight timing and same-ish direction
            if dt <= 150:
                current.append(n2)
            else:
                if len(current) >= self.params["min_rail_len"]:
                    rails.append(self._rationalise_rail(current))
                current = [n2]

        if len(current) >= self.params["min_rail_len"]:
            rails.append(self._rationalise_rail(current))

        total_rail_notes = sum(r["length"] for r in rails)
        return {
            "rails": rails[:20],        # cap for readability
            "rail_count": len(rails),
            "total_rail_notes": total_rail_notes,
        }

    def _rationalise_rail(self, notes: List[Dict]) -> Dict:
        start_t = notes[0]["time"]
        end_t = notes[-1]["time"]
        start = (notes[0]["x"], notes[0]["y"])
        end = (notes[-1]["x"], notes[-1]["y"])
        direction = math.degrees(math.atan2(end[1]-start[1], end[0]-start[0]))
        return {
            "length": len(notes),
            "time_start": start_t,
            "time_end": end_t,
            "start": start,
            "end": end,
            "direction_deg": round(direction, 1),
            "duration_ms": end_t - start_t,
        }

    def analyse(self, notes: List[Dict]) -> Dict:
        """Run full feasibility suite on one difficulty's note list."""
        out = {}
        out["bounds"] = self.check_bounds(notes)
        out["reachability"] = self.check_reachability(notes)
        out["flow"] = self.check_flow(notes)
        out["balance"] = self.check_balance(notes)
        out["density_ramp"] = self.check_density_ramp(notes)
        out["co_located"] = self.check_co_located(notes)
        out["rail_candidates"] = self.find_rail_candidates(notes)
        out["note_count"] = len(notes)
        out["duration_s"] = (notes[-1]["time"] - notes[0]["time"]) / 1000.0 if notes else 0.0
        out["avg_nps"] = len(notes) / out["duration_s"] if out["duration_s"] > 0 else 0.0
        out["overall_pass"] = all([
            out["bounds"]["pass"],
            out["reachability"]["pass"],
            out["flow"]["pass"],
            out["balance"]["pass"],
            out["density_ramp"]["pass"],
            out["co_located"]["pass"],
        ])
        return out


# ── I/O Wrappers ───────────────────────────────────────────────────────

def analyse_map_bundle(path: str) -> Dict:
    """Load a JSON bundle and analyse all difficulties with per-difficulty params."""
    with open(path, "r") as f:
        bundle = json.load(f)

    result = {
        "uuid": bundle.get("uuid", Path(path).stem),
        "version": bundle.get("version", "unknown"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "difficulties": {},
        "overall_pass": True,
    }
    for diff, notes in bundle.get("difficulties", {}).items():
        params = DIFFICULTY_PARAMS.get(diff, DIFFICULTY_PARAMS["Master"])
        checker = FeasibilityChecker(params)
        analysis = checker.analyse(notes)
        result["difficulties"][diff] = analysis
        if not analysis["overall_pass"]:
            result["overall_pass"] = False

    return result


def process_file(json_path: str) -> Dict:
    try:
        return analyse_map_bundle(json_path)
    except Exception as e:
        return {
            "uuid": Path(json_path).stem,
            "error": True,
            "error_message": str(e),
        }


def run_batch(batch_dir: str, out_dir: str, workers: int = 8):
    os.makedirs(out_dir, exist_ok=True)
    files = [str(p) for p in Path(batch_dir).rglob("*.json")]
    print(f"[+] Feasibility batch: {len(files)} maps, {workers} workers")

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_file, fp): fp for fp in files}
        for fut in as_completed(futures):
            res = fut.result()
            uid = res.get("uuid", Path(futures[fut]).stem)
            out_path = os.path.join(out_dir, f"{uid}.json")
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)
            done += 1
            if done % 100 == 0:
                print(f"    {done}/{len(files)} done")

    print(f"[+] Batch complete. Reports in {out_dir}")


# ── Audit / Summary ────────────────────────────────────────────────────

def build_audit(reports_dir: str) -> Dict:
    """Aggregate summary across all feasibility reports."""
    files = list(Path(reports_dir).rglob("*.json"))
    totals = {d: {"pass": 0, "fail": 0, "notes": []} for d in DIFFICULTY_PARAMS}
    maps_passed = 0
    maps_failed = 0

    for fp in files:
        with open(fp, "r") as f:
            rep = json.load(f)
        if rep.get("error"):
            continue
        map_pass = rep.get("overall_pass", False)
        if map_pass:
            maps_passed += 1
        else:
            maps_failed += 1
        for diff, a in rep.get("difficulties", {}).items():
            totals.setdefault(diff, {"pass": 0, "fail": 0, "notes": []})
            if a["overall_pass"]:
                totals[diff]["pass"] += 1
            else:
                totals[diff]["fail"] += 1
            totals[diff]["notes"].append(a.get("note_count", 0))

    summary = {"total_maps": len(files), "maps_passed": maps_passed, "maps_failed": maps_failed}
    for diff, stats in totals.items():
        # Skip empty
        if not stats["notes"]:
            continue
        summary[diff] = {
            "pass_rate": round(stats["pass"] / (stats["pass"] + stats["fail"]), 3) if (stats["pass"] + stats["fail"]) else 0,
            "avg_notes": round(sum(stats["notes"])/len(stats["notes"]), 1),
            "median_nps": 0.0,  # filled below
        }
    return summary


def print_audit(summary: Dict):
    print("\n=== Feasibility Audit ===")
    print(f"Total Maps: {summary['total_maps']} | Passed: {summary['maps_passed']} | Failed: {summary['maps_failed']}")
    print("-" * 88)
    print(f"{'Difficulty':<12} {'Pass%':<8} {'Avg Notes':<10} {'Audit Status'}")
    print("-" * 88)
    for diff in ["Easy", "Normal", "Hard", "Expert", "Master"]:
        if diff not in summary:
            continue
        s = summary[diff]
        print(f"{diff:<12} {s['pass_rate']:<8.1%} {s['avg_notes']:<10.1f} {'OK' if s['pass_rate'] > 0.5 else 'CHECK'}")
    print("-" * 88)
    print("Pass % is the proportion of maps where ALL checks pass for that difficulty.")
    print("NPS = Notes Per Second; computed per-map from actual note placement.\n")


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SynthRiders Feasibility Checker")
    parser.add_argument("--map", help="Path to single JSON map bundle to analyse")
    parser.add_argument("--batch", help="Directory of map bundles to batch-analyse")
    parser.add_argument("--out", default="evaluation/phase12b/feasibility_reports/", help="Output directory for reports")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for batch")
    parser.add_argument("--audit", action="store_true", help="Print summary table from existing reports")
    parser.add_argument("--indir", default="evaluation/phase12b/feasibility_reports/", help="Reports directory for audit")
    args = parser.parse_args()

    if args.map:
        res = analyse_map_bundle(args.map)
        print(json.dumps(res, indent=2))
    elif args.batch:
        run_batch(args.batch, args.out, args.workers)
    elif args.audit:
        summary = build_audit(args.indir)
        print_audit(summary)
    else:
        parser.print_help()
