"""Restore original files from the adjacent split gzip archives."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

archive_dir = Path(__file__).resolve().parent
repo = archive_dir.parent.parent
for entry in json.loads((archive_dir / 'manifest.json').read_text()):
    destination = repo / entry['path']
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with tempfile.TemporaryFile() as compressed:
        for part in entry['parts']:
            with (archive_dir / part).open('rb') as source:
                shutil.copyfileobj(source, compressed)
        compressed.seek(0)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                temporary = Path(output.name)
                with gzip.GzipFile(fileobj=compressed, mode='rb') as source:
                    while chunk := source.read(4 * 1024 * 1024):
                        digest.update(chunk)
                        output.write(chunk)
            if digest.hexdigest() != entry['sha256'] or temporary.stat().st_size != entry['size']:
                raise ValueError(f"Verification failed: {entry['path']}")
            os.replace(temporary, destination)
            print(f"Restored and verified {entry['path']}")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
