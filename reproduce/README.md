# Statistical reproduction

```sh
uv run --no-sync dafc reproduce
```

The verifier checks the publication-input hashes and all 1,120 outcome rows,
then recomputes eight policy summaries and six paired, task-cluster bootstrap
comparisons with 20,000 resamples and the original seeds. The statistical
functions retain the C588 analyzer's implementation; `SOURCE.json` records
its original file identity. `python reproduce/verify.py` is the repository
entrypoint.

## Published inputs

Public outcome rows form an identified derivative: absolute attempt paths
are replaced by relative paths, and machine-binding and infrastructure
diagnostic objects are omitted. `redactions.json` records the transformation.
The original source and raw records retain their separate identities.

## Original-attempt verification

```sh
uv run --no-sync dafc reproduce --raw-root /path/to/formal_android_atomic_c587_v3
```

This mode requires the original attempt directory, approximately 853 MB.
It checks `raw_manifest.json`, image-receipt hashes, the single permitted
infrastructure retry, one outcome per job and the raw-to-public projection.
The default mode reconstructs statistics from the packaged public inputs;
this additional mode verifies their original-attempt provenance.

New experiments record the current source and inputs under a distinct run
identity. Browser and Android setup is described in
[resources](../docs/RESOURCES.md).

The current public edition omits recorded model-message bodies. Its integrity
indexes bind the published records; recorded statistical outcomes are retained.
