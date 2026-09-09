# Echorin

Echorin is a modular 2D sensing simulator that separates target ground truth
from synthetic radar observations. It advances moving targets, generates noisy
delayed echoes, applies matched filtering and range processing, and detects
returns with fixed and CA-CFAR thresholds. A PySide6/PyQtGraph desktop interface
shows the simulated world and the sensor's range products.

```text
World truth -> propagation + noise -> matched filter -> range profile -> CFAR
     |                                                               |
     +---------------- optional GUI truth overlay -------------------+
```

## Installation

Echorin requires Python 3.12 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run

```bash
python main.py
# or, after installation
echorin
```

Run the automated suite with:

```bash
python -m pytest
```

## Current scope

The implemented release covers the specification through Milestone 5: project
foundation, world kinematics, basic GUI, synthetic radar echoes, matched-filter
range processing, fixed-threshold detection, and CA-CFAR. Tracking, Doppler,
and acoustic sonar processing are intentionally reserved for later milestones.

## Simulation assumptions

Echorin is an educational simulator, not a high-fidelity electromagnetic or
acoustic propagation model. It uses configurable two-way delay, simplified
amplitude decay, and additive white Gaussian noise. Ground truth is never passed
to detection logic; it is used only to synthesize observations and optionally
visualize/evaluate results.
