#!/usr/bin/env python3
"""Sanitize compile_commands.json entries that invoke nvcc for IDE tooling."""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import re
import subprocess
from typing import Iterable, List

_CommandLineToArgvW = ctypes.windll.shell32.CommandLineToArgvW
_CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
_CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
_LocalFree = ctypes.windll.kernel32.LocalFree

_NVCC_PATTERN = re.compile(r"nvcc(?:\.exe)?", re.IGNORECASE)
_ARCH_PATTERN = re.compile(r"sm_(\d+)")


def _split_windows_command(command: str) -> List[str]:
    argc = ctypes.c_int()
    argv = _CommandLineToArgvW(command, ctypes.byref(argc))
    if not argv:
        raise ctypes.WinError()
    try:
        return [argv[i] for i in range(argc.value)]
    finally:
        _LocalFree(argv)


def _normalize(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    return value


def _extract_architectures(tokens: Iterable[str]) -> List[str]:
    archs: List[str] = []
    for token in tokens:
        match = _ARCH_PATTERN.search(token)
        if match:
            archs.append(f"sm_{match.group(1)}")
    return sorted(set(archs))


def _sanitize_nvcc_command(tokens: List[str]) -> List[str]:
    if not tokens:
        return tokens

    nvcc_path = Path(tokens[0])
    cuda_root = nvcc_path.parent.parent

    include_dirs: List[str] = []
    system_dirs: List[str] = []
    defines: List[str] = []
    std_flag = "c++17"
    source: str | None = None

    i = 1
    while i < len(tokens):
        token = tokens[i]
        if token == "-isystem":
            if i + 1 < len(tokens):
                system_dirs.append(_normalize(tokens[i + 1]))
                i += 1
        elif token.startswith("-isystem") and len(token) > len("-isystem"):
            system_dirs.append(_normalize(token[len("-isystem") :]))
        elif token.startswith("-I"):
            include_dirs.append(_normalize(token[2:]))
        elif token.startswith("-D"):
            defines.append(token[2:])
        elif token.startswith("-std="):
            std_flag = token.split("=", 1)[1]
        elif token == "-c":
            if i + 1 < len(tokens):
                source = _normalize(tokens[i + 1])
                i += 1
        elif token in {"-x", "cu", "-forward-unknown-to-host-compiler"}:
            pass
        elif token.startswith("-Xcompiler"):
            pass
        elif token.startswith("--generate-code"):
            pass
        elif token == "-o":
            i += 1
        i += 1

    archs = _extract_architectures(tokens)

    clang_cmd: List[str] = ["clang++", "-x", "cuda", f"--cuda-path={cuda_root}"]
    clang_cmd.append(f"-std={std_flag}")
    for arch in archs:
        clang_cmd.append(f"--cuda-gpu-arch={arch}")
    clang_cmd.append("-fdiagnostics-absolute-paths")

    for define in defines:
        clang_cmd.append(f"-D{define}")
    for inc in include_dirs:
        clang_cmd.append(f"-I{inc}")
    for inc in system_dirs:
        clang_cmd.extend(["-isystem", inc])

    clang_cmd.append("-fsyntax-only")
    if source:
        clang_cmd.append(source)

    return clang_cmd


def _sanitize_entry(entry: dict) -> tuple[bool, dict]:
    updated = False
    result = dict(entry)

    if "arguments" in entry and entry["arguments"]:
        args = entry["arguments"]
        if _NVCC_PATTERN.search(args[0]):
            new_args = _sanitize_nvcc_command(list(args))
            result["arguments"] = new_args
            result.pop("command", None)
            updated = True
    else:
        command = entry.get("command")
        if command and _NVCC_PATTERN.search(command):
            tokens = _split_windows_command(command)
            new_tokens = _sanitize_nvcc_command(tokens)
            result["command"] = subprocess.list2cmdline(new_tokens)
            updated = True
    return updated, result


def sanitize_database(input_path: Path, output_path: Path) -> bool:
    entries = json.loads(input_path.read_text())
    changed = False
    sanitized: List[dict] = []

    for entry in entries:
        updated, new_entry = _sanitize_entry(entry)
        changed = changed or updated
        sanitized.append(new_entry)

    output_path.write_text(json.dumps(sanitized, indent=2) + "\n")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sanitize compile_commands.json for clang tooling."
    )
    parser.add_argument("input", type=Path, help="Source compile_commands.json")
    parser.add_argument(
        "-o", "--output", type=Path, help="Destination file (defaults to input path)"
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or args.input
    if not input_path.exists():
        raise SystemExit(f"Input compile_commands not found: {input_path}")

    changed = sanitize_database(input_path, output_path)
    if changed:
        print(f"Sanitized compile_commands written to {output_path}")
    else:
        print(f"No nvcc entries found; {output_path} written unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
