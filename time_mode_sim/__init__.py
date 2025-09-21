"""
Time-Mode Computation Simulation Framework
"""

from .blocks import (
    ChargePump,
    CompletionDetector,
    DigitalToTimeConverter,
    FixedWidthPulseGenerator,
    MonostableMultivibrator,
    SRLatch,
    TimeDomainSubtractor,
    TimeToDigitalConverter,
)
from .core import CurrentSource, DifferentialTimeSignal, SignalLevel, TimeSignal
from .neural import (
    ConvolutionalTimeLayer,
    RecurrentTimeLayer,
    ReLUActivation,
    SoftminActivation,
    ThresholdActivation,
    TimeActivation,
    TimeAutoencoder,
    TimeNeuralLayer,
    TimeNeuralNetwork,
)
from .visualization import (
    NetworkVisualizer,
    PerformanceAnalyzer,
    SignalVisualizer,
    TimingDiagram,
    WaveformAnalyzer,
)
from .vmm import ChainedVMM, PipelinedVMM, TimeVMM

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
