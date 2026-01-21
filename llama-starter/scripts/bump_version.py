#!/usr/bin/env python3
"""Version bump utility for iterative releases."""

import re
import sys
from pathlib import Path
from typing import Literal

VersionType = Literal["major", "minor", "patch"]

def get_current_version() -> tuple[int, int, int]:
    """Read current version from __init__.py"""
    init_file = Path("src/llama_rag/__init__.py")
    content = init_file.read_text()
    match = re.search(r'__version__ = "(\d+)\.(\d+)\.(\d+)"', content)
    if not match:
        raise ValueError("Version not found in __init__.py")
    return tuple(map(int, match.groups()))

def bump_version(version_type: VersionType) -> str:
    """Bump version based on type."""
    major, minor, patch = get_current_version()
    
    if version_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif version_type == "minor":
        minor += 1
        patch = 0
    elif version_type == "patch":
        patch += 1
    
    return f"{major}.{minor}.{patch}"

def update_version_in_files(new_version: str):
    """Update version in all relevant files."""
    files_to_update = {
        "src/llama_rag/__init__.py": r'__version__ = "\d+\.\d+\.\d+"',
        "setup.py": r'version="\d+\.\d+\.\d+"',
        "pyproject.toml": r'version = "\d+\.\d+\.\d+"',
    }
    
    for file_path, pattern in files_to_update.items():
        path = Path(file_path)
        if path.exists():
            content = path.read_text()
            # Determine the format based on file
            if "setup.py" in file_path:
                replacement = f'version="{new_version}"'
            elif "pyproject.toml" in file_path:
                replacement = f'version = "{new_version}"'
            else:
                replacement = f'__version__ = "{new_version}"'
            
            updated = re.sub(pattern, replacement, content)
            path.write_text(updated)
            print(f"✓ Updated {file_path}")

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ["major", "minor", "patch"]:
        print("Usage: python scripts/bump_version.py [major|minor|patch]")
        sys.exit(1)
    
    version_type = sys.argv[1]
    current = ".".join(map(str, get_current_version()))
    new_version = bump_version(version_type)
    
    print(f"Bumping version: {current} → {new_version}")
    update_version_in_files(new_version)
    print(f"\n✅ Version bumped to {new_version}")
    print(f"\nNext steps:")
    print(f"  1. Review changes: git diff")
    print(f"  2. Commit: git commit -am 'Bump version to {new_version}'")
    print(f"  3. Tag: git tag v{new_version}")
    print(f"  4. Build: python -m build")
    print(f"  5. Publish: twine upload dist/*llama_rag_lib-{new_version}*")

if __name__ == "__main__":
    main()
