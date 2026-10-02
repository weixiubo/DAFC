# Local development

Source: `~/project/DAFC`; virtual environment: `.venv` within this repository.
Inputs: `~/research/data/DAFC`; checkpoints: `~/research/models`; outputs:
`~/research/runs/DAFC/<run-id>`; historical originals: `~/research/data/<project>/evidence`.

All locations are configurable; see docs/RESOURCES.md. Install only the optional
groups needed by the process you are running. Basic development and verification
use the root Python 3.12 uv.lock. The original Android runtime used a separate
Python 3.11 environment; that environment is historical evidence, not a current
installation recipe. See the explicit compatibility boundary in the resource doc.
