# Compressed model and dataset files

These gzip archives are split into parts of at most 95 MB so they can be stored in regular Git. `manifest.json` lists each original path, size, SHA-256 checksum, and the ordered archive parts.

To restore all three originals, run from the repository root:

```sh
python3 'Project/Compressed Files/restore.py'
```

The script joins the parts, decompresses each file, checks its size and checksum, and writes it to its original path. Existing files at those paths are replaced after verification. The restored originals are ignored by Git; the archive parts are versioned.
