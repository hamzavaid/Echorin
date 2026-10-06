"""Print complete seed/configuration/accuracy/runtime evidence for v1.4."""

import json

from echorin.environment_benchmark import run_environment_benchmark

if __name__ == "__main__":
    print(json.dumps(run_environment_benchmark(), indent=2))
