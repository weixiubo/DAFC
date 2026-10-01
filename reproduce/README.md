# Offline reproduction

Run `python reproduce/verify.py` after installing the project. It hashes the
reviewed publication inputs, checks all 1120 outcome rows, and recomputes the
8 policy summaries and 6 paired 20,000-resample task-cluster bootstrap intervals
with the original seeds. Statistical function bodies are copied unchanged from
the C588 analyzer; `SOURCE.json` records the original full-file digest.

The public rows are a separately identified **redacted derivative**: only the
absolute attempt-path field was changed to a relative path. Full machine binding
and infrastructure diagnostic objects are omitted. `redactions.json` records
every changed row. Original analysis, manifest, source and raw attempts retain
their exact bytes in the private archive; no old checksum was rewritten.

For full local raw verification run:
`python reproduce/verify.py --raw-root /path/to/formal_android_atomic_c587_v3`.
This checks every file against `raw_manifest.json`, image-receipt hashes, the
single permitted infrastructure retry, exactly one outcome per job, and equality
of the raw-to-public projection. The raw material is about 853 MB and is not
embedded in the package. Without it, the default verifier establishes result
reconstruction from the published derivative, not complete raw provenance.

No model experiment was rerun. The refactored source has a new identity and is
not claimed to have produced the recorded evidence. Generate a new manifest for
new runs; never substitute this source into an old source-bound manifest.
