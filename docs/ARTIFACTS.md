# Research artifacts

The [research archive](https://github.com/weixiubo/DAFC/releases/tag/research-artifacts-2026-10-17-r3)
provides source versions, published experiment records, supporting inputs,
final manuscript material and figures. Maintained source and installation
instructions are available on the main branch.

## Download and verification

Use Git and Python on Linux or WSL. Project execution uses the Python version
and dependencies specified by the repository. Select an empty destination
for a different archive edition.

```sh
git clone https://github.com/weixiubo/DAFC.git
cd DAFC
python3 scripts/handoff.py restore --root "$PWD/../DAFC-materials"
python3 scripts/handoff.py verify --root "$PWD/../DAFC-materials"
source "../DAFC-materials/activate.sh"
uv sync --locked
```

Downloads require no GitHub account. The verifier checks file paths, sizes,
SHA-256 identities, permissions, links and declared source-history references.
Repeated restoration reuses verified files. Modified files, damaged parts or
unexpected Git references stop restoration. `--offline` uses already
downloaded and verified parts.

## Publication scope

The archive contains 185,399 file entries. Published research records retain
task inputs, parsed actions or outcomes, numerical measurements, failures and
model-call accounting. Recorded prompt bodies, model replies, reasoning text,
internal manuscript reviews and machine-control records are excluded.
Event logs are public metric views rather than conversation-replay checkpoints.
Source templates and synthetic regression fixtures are part of the software.

The manifest describes the bytes of this public edition. Original identities
in provenance fields refer to the recorded source version; those fields do
not assert that excluded message bodies are distributed. File and source
revision correspondence is supplied with the retained histories.

## Source versions

Restoration creates standalone bare repositories with checked object storage
and explicit references. Manuscript drafting history and internal working
material are excluded. Final manuscript files are supplied separately.

```sh
git clone ../DAFC-materials/snapshot/source-history/research/repository.git historical-source
```

Historical source revisions describe their recorded interfaces. Use the main
branch and its catalog with the current public data edition.

## Validation and external resources

The project reproduction commands reconstruct recorded statistics from the
published data. They do not rerun the original model-experiment matrix or
replay excluded conversations. Required model, browser, emulator and service
conditions are documented in the repository's resource and installation guides.
New experiments use separate output directories and their own configuration.

## Android attempt records

The published attempt records and images retain their original outcomes and
geometry. Full model-message bodies and internal service-stop records are
excluded. After restoration and activation, verify their statistical projection:

```sh
uv run --no-sync dafc reproduce --raw-root "$DAFC_DATA_DIR/android-atomic-original"
```

The existing directory and option names are retained for compatibility.
`raw_manifest.json` binds the public attempt files and records prior identities
for edited files. Images retain their original SHA-256 values. The two pinned
LightGlue/SuperPoint weights are downloaded from the official upstream release.
