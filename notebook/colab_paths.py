"""Path setup for Google Colab (browser or Cursor/VS Code extension)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

DRIVE_FOLDER = "chest_xray_pneumonia_detection"
DRIVE_FOLDER_ID = "1--_EhLNau9vcpeH8gv1XxBD_brfMYQ16"
MARKER = Path("src") / "train.py"


def running_in_colab() -> bool:
    if os.environ.get("COLAB_GPU") or os.environ.get("COLAB_RELEASE_TAG"):
        return True
    if Path("/content").exists() and not Path("/Applications").exists():
        return True
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount_drive_safe() -> bool:
    """Mount Drive if needed. Skips gracefully when extension auth fails."""
    mydrive = Path("/content/drive/MyDrive")
    if mydrive.exists():
        try:
            next(mydrive.iterdir())
            print("Google Drive already mounted.")
            return True
        except StopIteration:
            pass

    try:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=False)
        print("Google Drive mounted.")
        return True
    except Exception as exc:
        print("drive.mount() failed (common in Cursor/VS Code Colab extension).")
        print("  -> Cmd+Shift+P → Colab: Mount Google Drive to Server")
        print(f"  Error: {exc}")
        return False


def _is_project_root(path: Path) -> bool:
    return (path / MARKER).exists()


def search_drive_for_project(
    start: Path = Path("/content/drive/MyDrive"),
    max_depth: int = 5,
) -> Path | None:
    """Find folder containing src/train.py under MyDrive."""
    if not start.exists():
        return None
    start = start.resolve()
    # Breadth-first, skip hidden and huge dirs
    skip_names = {".git", ".venv", "chest_xray", "train", "test", "val", "__pycache__"}
    queue: list[tuple[Path, int]] = [(start, 0)]
    while queue:
        folder, depth = queue.pop(0)
        if _is_project_root(folder):
            return folder
        if depth >= max_depth:
            continue
        try:
            children = sorted(folder.iterdir())
        except (OSError, PermissionError):
            continue
        for child in children:
            if not child.is_dir() or child.name.startswith(".") or child.name in skip_names:
                continue
            queue.append((child, depth + 1))
    return None


def list_folder(path: Path, indent: str = "  ") -> None:
    """Print folder contents for debugging."""
    if not path.exists():
        print(f"{indent}(folder does not exist)")
        return
    try:
        items = sorted(path.iterdir())[:25]
    except (OSError, PermissionError) as e:
        print(f"{indent}(cannot read: {e})")
        return
    if not items:
        print(f"{indent}(empty folder)")
        return
    for item in items:
        suffix = "/" if item.is_dir() else ""
        print(f"{indent}{item.name}{suffix}")
    if len(items) >= 25:
        print(f"{indent}...")


def find_project_root(drive_folder: str = DRIVE_FOLDER) -> Path:
    here = Path.cwd()
    candidates = [
        Path(f"/content/drive/MyDrive/.shortcut-targets-by-id/by-id/{DRIVE_FOLDER_ID}"),
        Path(f"/content/drive/MyDrive/{drive_folder}"),
        Path(f"/content/drive/MyDrive/{drive_folder}/{drive_folder}"),  # double-nested zip
        Path("/content/drive/MyDrive/Chext_X Pneumonia detection/chest_xray_pneumonia_detection"),
        Path(
            "/content/drive/MyDrive/OTU WINTER 2026/Project 3/"
            "Chext_X Pneumonia detection/chest_xray_pneumonia_detection"
        ),
        here.parent if here.name == "notebook" else here,
        here,
        Path("/content/chest_xray_pneumonia_detection"),
    ]
    for path in candidates:
        if _is_project_root(path):
            return path.resolve()

    found = search_drive_for_project()
    if found is not None:
        print(f"Auto-found project at: {found}")
        return found

    return candidates[0].resolve()


def setup_colab(drive_folder: str = DRIVE_FOLDER) -> dict[str, Path]:
    """Mount Drive (if possible), resolve paths, chdir to project."""
    if running_in_colab():
        mount_drive_safe()

    root = find_project_root(drive_folder)

    if not _is_project_root(root):
        print(f"\nCould not find src/train.py under project root.")
        print(f"Tried: {root}\nContents:")
        list_folder(root)
        print(f"\nMy Drive top level:")
        list_folder(Path("/content/drive/MyDrive"))
        raise FileNotFoundError(
            f"Upload the FULL project folder to Drive (must include src/).\n"
            f"Expected: My Drive/{drive_folder}/src/train.py\n"
            f"Tip: zip chest_xray_pneumonia_detection on your Mac, upload zip to Drive, unzip in Drive."
        )

    paths = {
        "PROJECT_ROOT": root,
        "DATA_ROOT": root / "data" / "raw" / "chest_xray",
        "FIGURES_DIR": root / "reports" / "figures",
        "MODELS_DIR": root / "models",
    }
    for key, d in paths.items():
        if key != "PROJECT_ROOT":
            d.mkdir(parents=True, exist_ok=True)
    (paths["DATA_ROOT"].parent).mkdir(parents=True, exist_ok=True)

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)
    return paths
