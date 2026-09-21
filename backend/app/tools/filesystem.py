"""read_file, list_directory, glob, grep, write_file, edit_file."""

import fnmatch
import subprocess
from pathlib import Path
from typing import Any

from .sandbox import MAX_FILE_SIZE, MAX_MATCHES, MAX_RESULTS, ToolError, resolve_inside_workdir


def read_file(tool_input: dict[str, Any], workdir: Path) -> dict[str, Any]:
    resolved = resolve_inside_workdir(tool_input["path"], workdir)
    content = resolved.read_text(encoding="utf-8")
    if len(content) > MAX_FILE_SIZE:
        return {"content": content[:MAX_FILE_SIZE], "truncated": True, "total_length": len(content)}
    return {"content": content}


def list_directory(tool_input: dict[str, Any], workdir: Path) -> dict[str, Any]:
    resolved = resolve_inside_workdir(tool_input.get("path", "."), workdir)
    entries = []
    for entry in sorted(resolved.iterdir()):
        if entry.name.startswith(".") or entry.name in ("node_modules", "__pycache__"):
            continue
        entries.append({"name": entry.name, "type": "directory" if entry.is_dir() else "file"})
    entries.sort(key=lambda e: (e["type"] != "directory", e["name"]))
    return {"path": str(resolved.relative_to(workdir)) or ".", "entries": entries}


def glob_files(tool_input: dict[str, Any], workdir: Path) -> dict[str, Any]:
    resolved = resolve_inside_workdir(tool_input.get("path", "."), workdir)
    pattern = tool_input["pattern"]
    files: list[str] = []
    truncated = False
    for candidate in sorted(resolved.rglob("*")):
        if "node_modules" in candidate.parts or "__pycache__" in candidate.parts:
            continue
        if not candidate.is_file():
            continue
        rel = candidate.relative_to(resolved)
        if not fnmatch.fnmatch(str(rel), pattern):
            continue
        if len(files) >= MAX_RESULTS:
            truncated = True
            break
        files.append(str(candidate.relative_to(workdir)))
    return {"files": files, **({"truncated": True} if truncated else {})}


def grep_files(tool_input: dict[str, Any], workdir: Path) -> dict[str, Any]:
    resolved = resolve_inside_workdir(tool_input.get("path", "."), workdir)
    args = ["grep", "-rn", "--color=never", "--exclude-dir=node_modules", "--exclude-dir=.git", "-E"]
    if tool_input.get("include"):
        args.append(f"--include={tool_input['include']}")
    args.extend([tool_input["pattern"], str(resolved)])

    proc = subprocess.run(args, cwd=workdir, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        raise ToolError(f"grep failed: {proc.stderr.strip()}")
    if not proc.stdout.strip():
        return {"matches": [], "message": "No matches found"}

    lines = proc.stdout.strip().split("\n")
    matches = []
    truncated = False
    for line in lines:
        if len(matches) >= MAX_MATCHES:
            truncated = True
            break
        file_part, _, rest = line.partition(":")
        line_no, _, content = rest.partition(":")
        try:
            matches.append({
                "file": str(Path(file_part).resolve().relative_to(workdir)),
                "line": int(line_no),
                "content": content,
            })
        except (ValueError, OSError):
            continue
    return {"matches": matches, **({"truncated": True, "total_matches": len(lines)} if truncated else {})}


def write_file(tool_input: dict[str, Any], workdir: Path) -> dict[str, Any]:
    resolved = resolve_inside_workdir(tool_input["path"], workdir)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    content = tool_input["content"]
    resolved.write_text(content, encoding="utf-8")
    return {"success": True, "path": str(resolved.relative_to(workdir)), "bytes_written": len(content.encode("utf-8"))}


def edit_file(tool_input: dict[str, Any], workdir: Path) -> dict[str, Any]:
    resolved = resolve_inside_workdir(tool_input["path"], workdir)
    content = resolved.read_text(encoding="utf-8")
    old_string = tool_input["old_string"]
    occurrences = content.count(old_string)
    if occurrences == 0:
        raise ToolError("old_string not found in file")
    if occurrences > 1:
        raise ToolError(f"old_string is ambiguous; found {occurrences} matches")
    resolved.write_text(content.replace(old_string, tool_input["new_string"]), encoding="utf-8")
    return {"success": True, "path": str(resolved.relative_to(workdir))}
