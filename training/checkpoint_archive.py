"""
Checkpoint Archival Utility

Local-only safety net: copies any checkpoint referenced in a forensic report
to training/checkpoints_archive/ before the report is finalized.
Never git-tracked, never committed — avoids CRLF corruption risk entirely.
"""

import re
import shutil
from pathlib import Path
from typing import List

ARCHIVE_DIR = Path("training/checkpoints_archive")
CHECKPOINT_PATTERN = re.compile(r'training[/\\\\]+models[/\\\\]+[\w\-]+(?:\.pt)?')


def archive_referenced_checkpoints(report_path: str) -> List[Path]:
    """
    Scan a report for checkpoint paths and copy any found to the local archive.
    
    Args:
        report_path: Path to the forensic/audit/diagnostic report file.
        
    Returns:
        List of archived checkpoint destination paths.
    """
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    
    text = Path(report_path).read_text(encoding="utf-8")
    archived = []
    
    for match in CHECKPOINT_PATTERN.finditer(text):
        src_str = match.group()
        src = Path(src_str)
        
        if not src.exists():
            continue
            
        # Destination: <checkpoint_stem>_<report_stem>.pt
        dst = ARCHIVE_DIR / f"{src.stem}_{Path(report_path).stem}.pt"
        
        if not dst.exists():
            shutil.copy2(src, dst)
            archived.append(dst)
            print(f"[checkpoint_archive] {src} -> {dst}")
    
    return archived


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        archive_referenced_checkpoints(sys.argv[1])
    else:
        print("Usage: python -m training.checkpoint_archive <report_path>")