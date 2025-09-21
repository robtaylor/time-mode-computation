"""
Vector-Matrix Multiplication (VMM) implementation for time-mode computation.
Supports both single-quadrant and four-quadrant operations.
"""

import numpy as np

from .blocks import ChargePump
from .core import CurrentSource, DifferentialTimeSignal, TimeSignal


class TimeVMM:
    """
    Time-domain Vector-Matrix Multiplier.
    Performs y = Wx where inputs/outputs are time-encoded.
    """

    def __init__(
        self,
        weights: np.ndarray,
        max_current: float = 1e-6,
        capacitance: float = 1e-12,
        vth: float = 0.5,
        phase_duration: float = 1.0,
    ):
        """
        Initialize VMM.

        Args:
            weights: Weight matrix (N_outputs x N_inputs)
            max_current: Maximum current for weight encoding
            capacitance: Integration capacitor value
            vth: Threshold voltage
            phase_duration: Duration of each computation phase
        """
        self.weights = weights
        self.n_outputs, self.n_inputs = weights.shape
        self.max_current = max_current
        self.capacitance = capacitance
        self.vth = vth
        self.phase_duration = phase_duration

        # Create current sources for weights
        self.current_sources = self._create_current_sources()

        # Create charge pumps for each output
        self.charge_pumps = [ChargePump(capacitance, vth) for _ in range(self.n_outputs)]

    def _create_current_sources(self) -> list[list[CurrentSource]]:
        """Create current sources from weight matrix."""
        sources = []
        for i in range(self.n_outputs):
            row_sources = []
            for j in range(self.n_inputs):
                # Map weight to current (normalized)
                weight_normalized = abs(self.weights[i, j])
                current = weight_normalized * self.max_current
                source = CurrentSource(current, self.max_current, self.vth)
                row_sources.append(source)
            sources.append(row_sources)
        return sources

    def compute_single_quadrant(self, input_signals: list[TimeSignal]) -> list[TimeSignal]:
        """
        Compute VMM for non-negative inputs and weights.

        Args:
            input_signals: List of time-encoded input signals

        Returns:
            List of time-encoded output signals
        """
        assert len(input_signals) == self.n_inputs, "Input dimension mismatch"

        output_signals = []

        for i in range(self.n_outputs):
            # Compute weighted sum: y[i] = sum(w[i,j] * x[j])
            weighted_sum = 0.0

            for j in range(self.n_inputs):
                # Get input value from pulse width
                input_value = input_signals[j].get_pulse_width() / input_signals[j].duration
                # Weight is already normalized in the weight matrix
                weight = self.weights[i, j]
                weighted_sum += weight * input_value

            # Ensure output is in valid range [0, 1]
            weighted_sum = max(0.0, min(1.0, weighted_sum))

            # Create output signal with pulse width encoding the result
            output_signal = TimeSignal.from_pulse_width(
                weighted_sum * self.phase_duration, self.phase_duration
            )

            output_signals.append(output_signal)

        return output_signals

    def compute_four_quadrant(
        self, input_signals: list[DifferentialTimeSignal]
    ) -> list[DifferentialTimeSignal]:
        """
        Compute VMM for signed inputs and weights using differential signaling.

        Args:
            input_signals: List of differential time-encoded input signals

        Returns:
            List of differential time-encoded output signals
        """
        output_signals = []

        for i in range(self.n_outputs):
            # Compute weighted sum for signed values
            weighted_sum = 0.0

            for j in range(self.n_inputs):
                # Get signed input value
                input_value = input_signals[j].to_signed_value()
                # Use signed weight
                weight = self.weights[i, j]
                weighted_sum += weight * input_value

            # Create differential output signal
            output_signal = DifferentialTimeSignal.from_signed_value(
                weighted_sum,
                max_value=self.n_inputs,  # Scale based on number of inputs
            )

            output_signals.append(output_signal)

        return output_signals


class PipelinedVMM:
    """
    Pipelined VMM implementation for higher throughput.
    Uses alternating phases for computation and output generation.
    """

    def __init__(self, vmm: TimeVMM, pipeline_depth: int = 2):
        """
        Initialize pipelined VMM.

        Args:
            vmm: Base VMM module
            pipeline_depth: Number of pipeline stages
        """
        self.vmm = vmm
        self.pipeline_depth = pipeline_depth
        self.pipeline_registers = [[] for _ in range(pipeline_depth)]
        self.current_stage = 0

    def process(self, input_signals: list[TimeSignal]) -> list[TimeSignal] | None:
        """
        Process inputs through pipeline.

        Args:
            input_signals: Input signals for current stage

        Returns:
            Output signals if pipeline is full, None otherwise
        """
        # Add input to current stage
        self.pipeline_registers[self.current_stage] = input_signals

        # Process if we have data
        if self.current_stage > 0:
            # Process previous stage
            prev_stage = (self.current_stage - 1) % self.pipeline_depth
            if self.pipeline_registers[prev_stage]:
                output = self.vmm.compute_single_quadrant(self.pipeline_registers[prev_stage])
            else:
                output = None
        else:
            output = None

        # Advance pipeline
        self.current_stage = (self.current_stage + 1) % self.pipeline_depth

        return output


class ChainedVMM:
    """
    Chain multiple VMMs for deep networks.
    Handles signal propagation between layers.
    """

    def __init__(self, vmm_layers: list[TimeVMM]):
        """
        Initialize chained VMM.

        Args:
            vmm_layers: List of VMM layers to chain
        """
        self.layers = vmm_layers
        self.n_layers = len(vmm_layers)

    def forward(
        self, input_signals: list[TimeSignal], include_intermediate: bool = False
    ) -> list[TimeSignal]:
        """
        Forward propagate through all layers.

        Args:
            input_signals: Input to first layer
            include_intermediate: Whether to return intermediate outputs

        Returns:
            Output from final layer (or all layers if include_intermediate)
        """
        current_signals = input_signals
        intermediate_outputs = []

        for layer in self.layers:
            current_signals = layer.compute_single_quadrant(current_signals)
            if include_intermediate:
                intermediate_outputs.append(current_signals)

        if include_intermediate:
            return intermediate_outputs
        else:
            return current_signals
