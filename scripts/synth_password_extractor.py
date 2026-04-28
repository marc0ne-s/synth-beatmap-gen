#!/usr/bin/env python3
"""
synth_password_extractor.py

Extracts the AES-256 ZIP password from a SynthRiders Quest APK/OBB.

Usage:
    python synth_password_extractor.py --obb <path/to/main.*.obb> [--synth <file.synth>]
    python synth_password_extractor.py --auto [--synth <file.synth>]

--auto mode:
    - Requires a Meta Quest connected via ADB
    - Pulls the OBB from /sdcard/Android/obb/com.kluge.SynthRiders/
    - Extracts data.unity3d
    - Searches for the [Miku Serializer] GameObject -> MonoBehaviour
    - Prints the ZipPassword field value

Dependencies: UnityPy, pyzipper (optional, for verification)
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional


DEFAULT_GAMEOBJECT_NAME = "[Miku Serializer]"


def find_adb() -> str:
    """Locate ADB binary."""
    candidates = [
        "/Applications/SideQuest.app/Contents/Resources/app.asar.unpacked/build/platform-tools/adb",
        "/usr/local/bin/adb",
        "/opt/homebrew/bin/adb",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    result = subprocess.run(["which", "adb"], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    raise RuntimeError("ADB not found. Install Android platform-tools or SideQuest.")


def get_quest_obb_path(adb: str) -> str:
    """Find the main OBB path on a connected Quest."""
    for pkg in [
        "com.kluge.SynthRiders",
        "com.kluge.SynthRiders-rjCOJGaVLq3dbfFhdrj0XQ",
    ]:
        result = subprocess.run(
            [adb, "shell", f"ls /sdcard/Android/obb/{pkg}/main.*.obb"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip().splitlines()[0].strip()
    raise RuntimeError("Could not find SynthRiders OBB on Quest. Is it installed and connected?")


def pull_obb(adb: str, remote_path: str, local_path: str) -> None:
    subprocess.run([adb, "pull", remote_path, local_path], check=True)


def extract_data_unity3d(obb_path: str, dest_dir: str) -> str:
    """Extract assets/bin/Data/data.unity3d from the OBB ZIP."""
    target = "assets/bin/Data/data.unity3d"
    with zipfile.ZipFile(obb_path, "r") as zf:
        zf.extract(target, dest_dir)
    return os.path.join(dest_dir, target)


def _looks_like_password(s: str) -> bool:
    """Heuristic: passwords have mixed case, digits, and symbols."""
    return (
        any(c.islower() for c in s)
        and any(c.isupper() for c in s)
        and any(c.isdigit() for c in s)
        and any(not c.isalnum() for c in s)
    )


def _extract_strings_from_raw(raw: bytes) -> list:
    """Parse raw MonoBehaviour bytes for length-prefixed ASCII strings."""
    results = []
    offset = 0
    while offset < len(raw) - 4:
        length = struct.unpack("<i", raw[offset : offset + 4])[0]
        if 8 <= length <= 64 and offset + 4 + length <= len(raw):
            s_bytes = raw[offset + 4 : offset + 4 + length]
            if all(32 <= b < 127 for b in s_bytes):
                candidate = s_bytes.decode("ascii", errors="ignore")
                if _looks_like_password(candidate):
                    results.append(candidate)
        offset += 1
    return results


def extract_password_from_unity3d(unity3d_path: str) -> Optional[str]:
    """
    Search for the Serializer MonoBehaviour in a Unity data.unity3d and
    return the ZipPassword field value.

    Algorithm:
      1. Find the [Miku Serializer] GameObject path_id in level1.
      2. Search ALL objects (even those UnityPy can't parse) for raw bytes
         containing a PPtr reference to that GameObject (file_id=0, path_id=...).
      3. For matching objects, parse raw bytes for length-prefixed strings.
      4. Return the first string that looks like a password.
    """
    import UnityPy

    env = UnityPy.load(unity3d_path)

    # Phase 1: find [Miku Serializer] GameObject path_id in level1
    target_go_path_id = None
    for asset in env.assets:
        if asset.name != "level1":
            continue
        for obj in asset.objects.values():
            try:
                data = obj.read()
                if type(data).__name__ == "GameObject":
                    name = getattr(data, "m_Name", "")
                    if name == DEFAULT_GAMEOBJECT_NAME:
                        target_go_path_id = obj.path_id
                        print(f"[+] Found GameObject '{name}' in {asset.name} path_id={obj.path_id}")
                        break
            except Exception:
                pass
        if target_go_path_id:
            break

    if not target_go_path_id:
        raise RuntimeError(f"GameObject '{DEFAULT_GAMEOBJECT_NAME}' not found in level1.")

    # Phase 2: search all objects for raw bytes referencing this GameObject
    # A PPtr in Unity binary is: m_FileID (int32=0) + m_PathID (int64)
    needle = struct.pack("<iq", 0, target_go_path_id)

    candidates = []
    for asset in env.assets:
        for obj in asset.objects.values():
            try:
                raw = obj.get_raw_data()
            except Exception:
                continue
            if needle not in raw:
                continue

            # This object references our GameObject. Try to read type.
            try:
                data = obj.read()
                typename = type(data).__name__
            except Exception:
                typename = f"type_{obj.type}"

            strings = _extract_strings_from_raw(raw)
            if strings:
                for s in strings:
                    candidates.append({
                        "asset": asset.name,
                        "path_id": obj.path_id,
                        "type": typename,
                        "password": s,
                    })
                print(f"[+] Candidate in {asset.name} path_id={obj.path_id} type={typename}: {strings[0]}")

    if not candidates:
        raise RuntimeError("No MonoBehaviour with password found.")

    return candidates[0]["password"]


def verify_password(synth_path: str, password: str) -> bool:
    """Verify password against an encrypted .synth file."""
    try:
        import pyzipper
    except ImportError:
        print("[!] pyzipper not installed; cannot verify password.")
        return False

    try:
        with pyzipper.AESZipFile(synth_path, "r") as zf:
            zf.setpassword(password.encode("utf-8"))
            zf.read(zf.namelist()[0])
            return True
    except Exception as e:
        print(f"[!] Verification failed: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SynthRiders ZIP password from APK/OBB")
    parser.add_argument("--auto", action="store_true", help="Auto-detect connected Quest and pull OBB")
    parser.add_argument("--obb", help="Path to OBB file")
    parser.add_argument("--synth", help="Optional .synth file to verify password against")
    parser.add_argument("--output", "-o", help="Write password to file")
    args = parser.parse_args()

    if args.auto:
        adb = find_adb()
        print(f"[+] ADB found: {adb}")
        remote_obb = get_quest_obb_path(adb)
        print(f"[+] OBB on Quest: {remote_obb}")
        local_obb = "/tmp/synth_obb_main.obb"
        print(f"[+] Pulling OBB to {local_obb} ...")
        pull_obb(adb, remote_obb, local_obb)
        obb_path = local_obb
    else:
        if not args.obb:
            parser.error("--obb required (or use --auto)")
        obb_path = args.obb

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"[+] Extracting data.unity3d from OBB ...")
        unity3d_path = extract_data_unity3d(obb_path, tmpdir)
        print(f"[+] Loaded: {unity3d_path}")

        print("[+] Searching for Serializer MonoBehaviour ...")
        password = extract_password_from_unity3d(unity3d_path)

    if password:
        print(f"\n[+] Extracted password: {password}")
        if args.synth:
            if verify_password(args.synth, password):
                print(f"[+] Verified against {args.synth}")
            else:
                print(f"[!] FAILED verification against {args.synth}")
        if args.output:
            with open(args.output, "w") as f:
                f.write(password)
            print(f"[+] Wrote password to {args.output}")
    else:
        print("[!] Password not found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
