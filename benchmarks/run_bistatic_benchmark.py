"""Print v1.5 moving Radar/Sonar TX/RX evidence, without generating repo output."""

import json

from echorin.bistatic_benchmark import run_bistatic_benchmark

if __name__ == "__main__":
    print(json.dumps(run_bistatic_benchmark(), indent=2))
