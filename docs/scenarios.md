# Deterministic sensing scenarios

## v1.2 same-range angular resolution

Run `python benchmarks/run_array_benchmark.py` from an installed source tree.
The script reports its seed, Radar configuration, target range/bearings, measured
bearings, range and angle bin widths, array size, pulse count, and stage timings
as JSON. Two point targets are placed at 300 m and bearings -0.45 and +0.45
radians. Both occupy the same range bin. At seed 7 and receiver noise sigma
0.001, each reported bearing must be within one 1-degree scan bin of its
configured position. The target identifiers are used only to construct truth;
the production array frame and detector receive no identifier or truth bearing.

This is a high-SNR educational far-field example, not a general angular
resolution guarantee. Array aperture, SNR, separation, and ULA ambiguity affect
other cases.
