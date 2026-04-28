#!/usr/bin/env python3
"""
synth_decryptor.py

Standalone tool to read/write SynthRiders .synth files.

SynthRiders uses ZIP archives with a .synth extension.
- Community maps (synthriderz.com): plain ZIP, no password
- Official Quest maps: AES-256 encrypted ZIP (WinZip AE-2 extension)

This script handles both formats.
"""

import argparse
import json
import os
import struct
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional


class SynthFileError(RuntimeError):
    pass


class EncryptedSynthError(SynthFileError):
    def __init__(self, msg: str, salt: Optional[bytes] = None) -> None:
        super().__init__(msg)
        self.salt = salt


def _decode_json(raw: bytes) -> dict:
    """Decode raw bytes to JSON, handling UTF-16 BOM."""
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return json.loads(raw.decode("utf-16"))
    return json.loads(raw.decode("utf-8"))


def _is_aes_encrypted(zf: zipfile.ZipFile) -> bool:
    """Check if any entry in the ZIP uses WinZip AES encryption (compress_type=99)."""
    for info in zf.infolist():
        if info.compress_type == 99:
            return True
    return False


def _extract_aes_salt(info: zipfile.ZipInfo, zf: zipfile.ZipFile) -> Optional[bytes]:
    """Extract the AES salt from the local file header of an encrypted entry."""
    # The salt is stored immediately after the local file header, before the encrypted data.
    # Salt sizes: AES-128 = 8 bytes, AES-192 = 12 bytes, AES-256 = 16 bytes
    # We need to read the raw file to locate the salt.
    with zf.open(info.filename) as fh:
        # Read enough to cover salt + password verifier (2 bytes)
        # For AES-256: 16 bytes salt + 2 bytes verifier = 18 bytes
        header = fh.read(32)
        if len(header) >= 18:
            return header[:16]
    return None


def _guess_password_verifier(info: zipfile.ZipInfo) -> Optional[bytes]:
    """For AE-2, the password verifier is not used (set to 0x0000).
    For AE-1, it would be a 2-byte value derived from the password.
    We can read it from the raw ZIP stream."""
    # This requires reading past the local file header, which is tricky with standard zipfile.
    # We'll return None for now and let brute-force handle it.
    return None


def read_synth(synth_path: str, password: Optional[str] = None) -> dict:
    """
    Read a .synth file and return its JSON content.

    For plain ZIPs, returns the first .json entry found.
    For encrypted ZIPs, *password* must be provided.
    Mixed archives (some entries encrypted, some not) are supported:
    unencrypted JSON entries are read without a password.
    """
    # Detect encryption using standard zipfile (pyzipper can't list compress_type=99)
    with zipfile.ZipFile(synth_path, "r") as zf_check:
        encrypted = _is_aes_encrypted(zf_check)

    if encrypted and password:
        try:
            import pyzipper
        except ImportError:
            raise SynthFileError(
                "Reading AES-encrypted ZIPs requires 'pyzipper'. "
                "Install it with: pip install pyzipper"
            )
        with pyzipper.AESZipFile(synth_path, "r") as zf:
            zf.setpassword(password.encode("utf-8"))
            # Prefer track.data.json (actual beatmap metadata) over synthriderz.meta.json
            for info in zf.infolist():
                if info.filename == "track.data.json":
                    raw = zf.read(info.filename)
                    return _decode_json(raw)
            for info in zf.infolist():
                if info.filename.endswith(".json"):
                    raw = zf.read(info.filename)
                    return _decode_json(raw)
            # Fallback: first entry
            info = zf.infolist()[0]
            raw = zf.read(info.filename)
            return _decode_json(raw)
    else:
        with zipfile.ZipFile(synth_path, "r") as zf:
            if password:
                zf.setpassword(password.encode("utf-8"))

            # Prefer unencrypted JSON entries first (no password needed)
            for info in zf.infolist():
                if info.filename == "track.data.json" and info.compress_type != 99:
                    raw = zf.read(info.filename)
                    return _decode_json(raw)
            for info in zf.infolist():
                if info.filename.endswith(".json") and info.compress_type != 99:
                    raw = zf.read(info.filename)
                    return _decode_json(raw)

            # If no unencrypted JSON, try encrypted JSON entries
            for info in zf.infolist():
                if info.filename.endswith(".json"):
                    if info.compress_type == 99 and not password:
                        salt = _extract_aes_salt(info, zf)
                        raise EncryptedSynthError(
                            "File is AES-256 encrypted. Provide --password to decrypt.",
                            salt=salt,
                        )
                    raw = zf.read(info.filename)
                    return _decode_json(raw)

            # Fallback: first entry (encrypted or not)
            info = zf.infolist()[0]
            if info.compress_type == 99 and not password:
                salt = _extract_aes_salt(info, zf)
                raise EncryptedSynthError(
                    "File is AES-256 encrypted. Provide --password to decrypt.",
                    salt=salt,
                )
            raw = zf.read(info.filename)
            return _decode_json(raw)


def write_synth(
    synth_path: str,
    data: dict,
    password: Optional[str] = None,
    audio_path: Optional[str] = None,
    cover_path: Optional[str] = None,
) -> None:
    """
    Write a .synth file from JSON data.

    If *password* is provided, the ZIP will be AES-256 encrypted using Ionic.Zip
    compatible settings (WinZip AE-2, compress_type=99).

    If *audio_path* or *cover_path* are provided, they are bundled into the ZIP.
    """
    # Use temp file to avoid corrupting existing file on error
    tmp_path = synth_path + ".tmp"

    # Ionic.Zip / WinZip AES requires the pyzipper library or manual ZIP construction.
    # Standard zipfile cannot write AES-encrypted entries.
    # We try to use pyzipper if available; otherwise we write a plain ZIP.
    if password:
        try:
            import pyzipper
        except ImportError:
            raise SynthFileError(
                "Writing AES-encrypted ZIPs requires 'pyzipper'. "
                "Install it with: pip install pyzipper"
            )

        with pyzipper.AESZipFile(
            tmp_path, "w", compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES
        ) as zf:
            zf.setpassword(password.encode("utf-8"))
            zf.writestr("song.json", json.dumps(data, indent=2).encode("utf-8"))
            if audio_path:
                zf.write(audio_path, os.path.basename(audio_path))
            if cover_path:
                zf.write(cover_path, os.path.basename(cover_path))
    else:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("song.json", json.dumps(data, indent=2).encode("utf-8"))
            if audio_path:
                zf.write(audio_path, os.path.basename(audio_path))
            if cover_path:
                zf.write(cover_path, os.path.basename(cover_path))

    os.replace(tmp_path, synth_path)


def brute_force_password(synth_path: str, wordlist_path: Optional[str] = None) -> Optional[str]:
    """
    Try to brute-force the password of an encrypted .synth file.

    If *wordlist_path* is provided, read passwords from that file (one per line).
    Otherwise use a small built-in wordlist.
    """
    built_in = [
        "SynthRiders", "synthriders", "SynthRiders2024", "SynthRiders2025",
        "kluge", "Kluge", "KLUGE", "Synth", "synth", "password", "Password",
        "12345678", "00000000", "unity", "Unity", "Ionic", "ionic", "zip",
        "Zip", "encrypt", "Encrypt", "secret", "Secret", "game", "Game",
        "vr", "VR", "quest", "Quest", "meta", "Meta", "oculus", "Oculus",
        "SVR", "svr", "SynthRidersUC", "SynthRidersCustom", "customsongs",
        "CustomSongs", "beatmap", "Beatmap", "Songs", "songs", "Custom",
        "custom", "Map", "map", "Note", "note", "Rider", "rider",
        "SynthRidersOST", "OST", "ost", "Music", "music", "Audio", "audio",
        "Track", "track", "Level", "level", "Stage", "stage", "Beat", "beat",
    ]

    if wordlist_path:
        with open(wordlist_path, "r", encoding="utf-8", errors="ignore") as f:
            passwords = [line.strip() for line in f if line.strip()]
    else:
        passwords = built_in

    print(f"Testing {len(passwords)} passwords on {synth_path} ...")
    try:
        import pyzipper
    except ImportError:
        raise SynthFileError(
            "Brute-forcing AES-encrypted ZIPs requires 'pyzipper'. "
            "Install it with: pip install pyzipper"
        )

    for pwd in passwords:
        try:
            with pyzipper.AESZipFile(synth_path, "r") as zf:
                zf.setpassword(pwd.encode("utf-8"))
                zf.read(zf.namelist()[0])
                print(f"SUCCESS: password = {pwd}")
                return pwd
        except Exception:
            pass

    print("Password not found.")
    return None


def show_info(synth_path: str) -> None:
    """Display metadata about a .synth file."""
    print(f"File: {synth_path}")
    print(f"Size: {os.path.getsize(synth_path):,} bytes")

    with zipfile.ZipFile(synth_path, "r") as zf:
        encrypted = _is_aes_encrypted(zf)
        print(f"Encrypted: {encrypted}")

        for info in zf.infolist():
            extra_hex = info.extra.hex() if info.extra else ""
            compress_type_name = "AES-256" if info.compress_type == 99 else str(info.compress_type)
            print(f"  {info.filename:40s}  {info.file_size:>10,}  {compress_type_name:>8s}  extra={extra_hex[:40]}")

            if info.compress_type == 99 and extra_hex:
                # Parse WinZip AES extra field
                try:
                    tag, size = struct.unpack("<HH", info.extra[:4])
                    if tag == 0x9901:
                        version, vendor, strength, method = struct.unpack("<HHBH", info.extra[4:11])
                        vendor_str = struct.pack("<H", vendor).decode("ascii", errors="ignore")
                        print(f"    -> WinZip AES extra field: version={version}, vendor={vendor_str}, strength={strength} ({'AES-128' if strength==1 else 'AES-192' if strength==2 else 'AES-256'}), method={method}")
                except Exception:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="SynthRiders .synth file decryptor/inspector")
    sub = parser.add_subparsers(dest="command", required=True)

    # info
    p_info = sub.add_parser("info", help="Show file metadata")
    p_info.add_argument("synth", help="Path to .synth file")

    # read
    p_read = sub.add_parser("read", help="Read JSON from .synth")
    p_read.add_argument("synth", help="Path to .synth file")
    p_read.add_argument("--password", "-p", help="Decryption password (for encrypted files)")
    p_read.add_argument("--output", "-o", help="Optional file to write JSON to")

    # write
    p_write = sub.add_parser("write", help="Write JSON into a .synth file")
    p_write.add_argument("json", help="Path to JSON file")
    p_write.add_argument("synth", help="Output .synth file path")
    p_write.add_argument("--password", "-p", help="Encrypt with this password")
    p_write.add_argument("--audio", "-a", help="Path to audio file to bundle")
    p_write.add_argument("--cover", "-c", help="Path to cover image to bundle")

    # brute-force
    p_bf = sub.add_parser("brute", help="Brute-force password")
    p_bf.add_argument("synth", help="Path to encrypted .synth file")
    p_bf.add_argument("--wordlist", "-w", help="Path to wordlist file")

    args = parser.parse_args()

    if args.command == "info":
        show_info(args.synth)

    elif args.command == "read":
        try:
            data = read_synth(args.synth, password=args.password)
        except EncryptedSynthError as e:
            print(f"ERROR: {e}")
            if e.salt:
                print(f"  AES salt (first entry): {e.salt.hex()}")
            sys.exit(1)

        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"Wrote JSON to {args.output}")
        else:
            print(json_str)

    elif args.command == "write":
        with open(args.json, "r", encoding="utf-8") as f:
            data = json.load(f)
        write_synth(
            args.synth,
            data,
            password=args.password,
            audio_path=args.audio,
            cover_path=args.cover,
        )
        print(f"Wrote {args.synth}")

    elif args.command == "brute":
        brute_force_password(args.synth, wordlist_path=args.wordlist)


if __name__ == "__main__":
    main()
