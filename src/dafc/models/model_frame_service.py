"""Start the Qwen smart-resized model-frame service with an explicit backend."""

from dafc.models import model_frame as runner
from dafc.models import service as service

SERVICE_VERSION = "remote-open-gui-grounder-v7-qwen-model-frame-service"


def main():
    return service.main(runner=runner, service_version=SERVICE_VERSION)


if __name__ == "__main__":
    raise SystemExit(main())
