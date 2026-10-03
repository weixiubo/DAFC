# Third-party notices

Project-owned code is licensed under Apache-2.0. Python dependencies retain
their upstream licenses; dependency groups and versions are declared in
[pyproject.toml](pyproject.toml) and `uv.lock`.

Browser and Android workflows use separately supplied MiniWoB, AndroidWorld,
browser-driver and emulator resources. Learned matching uses an explicit
LightGlue source revision and its associated weights. Model services use
separately obtained Qwen or UI-TARS checkpoints. The
[resource catalog](src/dafc/resource_catalog.json) records source locations,
revisions, license references and known checksums. Model weights, APK assets
and benchmark contents remain subject to their respective upstream terms.

The package includes the C587-v3 public outcome derivative and statistical
verification resources. The [reproduction guide](reproduce/README.md)
documents their provenance and the separately supplied original-attempt
inputs. Regression reference modules retain the identities listed in
[tests/reference/MANIFEST.json](tests/reference/MANIFEST.json).
