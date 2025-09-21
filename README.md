# Time-Mode Computation Simulation Framework

[![Test](https://github.com/robtaylor/time-mode-computation/actions/workflows/test.yml/badge.svg)](https://github.com/robtaylor/time-mode-computation/actions/workflows/test.yml)
[![Lint](https://github.com/robtaylor/time-mode-computation/actions/workflows/lint.yml/badge.svg)](https://github.com/robtaylor/time-mode-computation/actions/workflows/lint.yml)

A Python framework for simulating time-mode analog computation circuits and systems, based on recent research in ultra-low energy neuromorphic computing.

## Overview

Time-mode signal processing (TMSP) represents analog information as time intervals between digital events rather than voltage or current amplitudes. This approach combines the computational efficiency of analog processing with the noise immunity of digital signals, enabling ultra-low energy operation in advanced CMOS processes.

This framework provides:
- Signal representation classes for time-encoded values
- Basic building blocks (MSMV, FWPG, charge pumps, etc.)
- Vector-matrix multiplication (VMM) implementations
- Neural network architectures in time domain
- Visualization and performance analysis tools

## Installation

```bash
# Clone the repository
git clone https://github.com/robtaylor/time-mode-computation.git
cd time-mode-computation

# Install dependencies (using PDM)
pdm install

# Or using pip
pip install numpy matplotlib
```

## Quick Start

```python
from time_mode_sim import TimeSignal, TimeVMM
import numpy as np

# Create weight matrix
weights = np.array([[0.8, 0.2], [0.3, 0.7]])

# Initialize VMM module
vmm = TimeVMM(weights)

# Create input signals
inputs = [
    TimeSignal.from_analog_value(0.5, encoding='pulse_width'),
    TimeSignal.from_analog_value(0.7, encoding='pulse_width')
]

# Compute matrix multiplication
outputs = vmm.compute_single_quadrant(inputs)

# Convert back to analog
result = [sig.to_analog_value() for sig in outputs]
print(f"VMM result: {result}")
```

## Examples

### Basic VMM Operations
```bash
python examples/basic_vmm.py
```
Demonstrates:
- Single-quadrant VMM with positive weights/inputs
- Four-quadrant VMM with signed values
- Pipelined VMM for higher throughput

### Neural Networks
```bash
python examples/neural_network.py
```
Shows:
- Simple feedforward networks
- Multi-layer architectures
- Autoencoder implementation
- Performance analysis and energy metrics

## Core Components

### Signal Representation
- `TimeSignal`: Encodes analog values as pulse widths or edge timings
- `DifferentialTimeSignal`: Represents signed values using differential encoding
- `CurrentSource`: Models programmable current sources for weight storage

### Building Blocks
- `MonostableMultivibrator`: Time-domain multiplication via charge integration
- `FixedWidthPulseGenerator`: Maintains signal flow between stages
- `ChargePump`: Integrates currents over time
- `TimeDomainSubtractor`: XOR-based subtraction

### Neural Network Components
- `TimeVMM`: Vector-matrix multiplication in time domain
- `TimeNeuralNetwork`: Multi-layer neural network
- `SoftminActivation`: Winner-take-all based on timing
- `ReLUActivation`: Threshold-based activation

## Performance Metrics

The framework models energy efficiency based on capacitive switching:
- ~7 fJ/Op for large arrays (N>500) with I/O overhead
- Sub-1 fJ/Op potential for core operations
- POps/J regime achievable with optimization

## References

This implementation is based on concepts from:

1. Bavandpour et al. (2019) - "Energy-Efficient Time-Domain Vector-by-Matrix Multiplier"
2. Akgun (2020) - "Time-Mode Digit Classification Neural Network"
3. Akgun (2018) - "Asynchronous Pipelined Time-to-Digital Converter"
4. Various papers on time-mode analog computation in `/papers` directory

## Contributing

Contributions are welcome! Areas of interest:
- Additional activation functions
- Training algorithms for time-domain networks
- Hardware-aware optimizations
- Benchmarking against other frameworks

## License

MIT License - See LICENSE file for details

## Contact

For questions or collaborations, please open an issue on GitHub.