"""
Time-Mode Computation Simulation Framework
"""

from .core import (
    SignalLevel,
    TimeSignal,
    DifferentialTimeSignal,
    CurrentSource
)

from .blocks import (
    MonostableMultivibrator,
    FixedWidthPulseGenerator,
    TimeDomainSubtractor,
    TimeToDigitalConverter,
    DigitalToTimeConverter,
    ChargePump,
    CompletionDetector,
    SRLatch
)

from .vmm import (
    TimeVMM,
    PipelinedVMM,
    ChainedVMM
)

from .neural import (
    TimeActivation,
    SoftminActivation,
    ReLUActivation,
    ThresholdActivation,
    TimeNeuralLayer,
    TimeNeuralNetwork,
    ConvolutionalTimeLayer,
    RecurrentTimeLayer,
    TimeAutoencoder
)

from .visualization import (
    SignalVisualizer,
    WaveformAnalyzer,
    PerformanceAnalyzer,
    NetworkVisualizer,
    TimingDiagram
)

__version__ = "0.1.0"

__all__ = [
    # Core
    'SignalLevel',
    'TimeSignal',
    'DifferentialTimeSignal',
    'CurrentSource',
    # Blocks
    'MonostableMultivibrator',
    'FixedWidthPulseGenerator',
    'TimeDomainSubtractor',
    'TimeToDigitalConverter',
    'DigitalToTimeConverter',
    'ChargePump',
    'CompletionDetector',
    'SRLatch',
    # VMM
    'TimeVMM',
    'PipelinedVMM',
    'ChainedVMM',
    # Neural
    'TimeActivation',
    'SoftminActivation',
    'ReLUActivation',
    'ThresholdActivation',
    'TimeNeuralLayer',
    'TimeNeuralNetwork',
    'ConvolutionalTimeLayer',
    'RecurrentTimeLayer',
    'TimeAutoencoder',
    # Visualization
    'SignalVisualizer',
    'WaveformAnalyzer',
    'PerformanceAnalyzer',
    'NetworkVisualizer',
    'TimingDiagram'
]