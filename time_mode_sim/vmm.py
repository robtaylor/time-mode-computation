"""
Vector-Matrix Multiplication (VMM) implementation for time-mode computation.
Supports both single-quadrant and four-quadrant operations.
"""

import numpy as np
from typing import List, Optional, Tuple
from .core import TimeSignal, DifferentialTimeSignal, CurrentSource
from .blocks import ChargePump, MonostableMultivibrator, FixedWidthPulseGenerator


class TimeVMM:
    """
    Time-domain Vector-Matrix Multiplier.
    Performs y = Wx where inputs/outputs are time-encoded.
    """
    def __init__(self, weights: np.ndarray, max_current: float = 1e-6,
                 capacitance: float = 1e-12, vth: float = 0.5,
                 phase_duration: float = 1.0):
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
        self.charge_pumps = [
            ChargePump(capacitance, vth) for _ in range(self.n_outputs)
        ]

    def _create_current_sources(self) -> List[List[CurrentSource]]:
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

    def compute_single_quadrant(self, input_signals: List[TimeSignal]) -> List[TimeSignal]:
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
            # Reset charge pump
            self.charge_pumps[i].reset()

            # Phase I: Accumulate weighted inputs
            currents = []
            durations = []

            for j in range(self.n_inputs):
                # Get input pulse width (encodes value)
                duration = input_signals[j].get_pulse_width()
                # Get weight as current
                current = self.current_sources[i][j].get_current()

                currents.append(current)
                durations.append(duration)

            # Integrate to get output voltage
            final_voltage = self.charge_pumps[i].integrate(currents, durations)

            # Phase II: Generate output pulse
            # Calculate bias current (as in papers)
            total_current = sum(self.current_sources[i][j].current
                              for j in range(self.n_inputs))
            bias_current = self.n_inputs * self.max_current - total_current

            # Time to reach threshold with constant current
            if bias_current > 0:
                output_time = self.charge_pumps[i].time_to_threshold(
                    self.n_inputs * self.max_current
                )
            else:
                output_time = self.phase_duration

            # Ensure output is within phase duration
            output_time = min(output_time, self.phase_duration)

            # Create output signal (falling edge encodes value)
            output_signal = TimeSignal([
                (0.0, 1),
                (self.phase_duration - output_time, 0)
            ], self.phase_duration * 2)  # Double duration for two phases

            output_signals.append(output_signal)

        return output_signals

    def compute_four_quadrant(self, input_signals: List[DifferentialTimeSignal]
                            ) -> List[DifferentialTimeSignal]:
        """
        Compute VMM for signed inputs and weights using differential signaling.

        Args:
            input_signals: List of differential time-encoded input signals

        Returns:
            List of differential time-encoded output signals
        """
        # Split into positive and negative components
        pos_inputs = [sig.positive for sig in input_signals]
        neg_inputs = [sig.negative for sig in input_signals]

        # Create weight matrices for four-quadrant operation
        weights_pp = np.maximum(self.weights, 0)  # Positive weights, positive inputs
        weights_nn = np.maximum(self.weights, 0)  # Positive weights, negative inputs
        weights_pn = -np.minimum(self.weights, 0)  # Negative weights, positive inputs
        weights_np = -np.minimum(self.weights, 0)  # Negative weights, negative inputs

        # Create VMMs for each quadrant
        vmm_pp = TimeVMM(weights_pp, self.max_current, self.capacitance,
                        self.vth, self.phase_duration)
        vmm_nn = TimeVMM(weights_nn, self.max_current, self.capacitance,
                        self.vth, self.phase_duration)
        vmm_pn = TimeVMM(weights_pn, self.max_current, self.capacitance,
                        self.vth, self.phase_duration)
        vmm_np = TimeVMM(weights_np, self.max_current, self.capacitance,
                        self.vth, self.phase_duration)

        # Compute each quadrant
        out_pp = vmm_pp.compute_single_quadrant(pos_inputs)
        out_nn = vmm_nn.compute_single_quadrant(neg_inputs)
        out_pn = vmm_pn.compute_single_quadrant(pos_inputs)
        out_np = vmm_np.compute_single_quadrant(neg_inputs)

        # Combine outputs
        output_signals = []
        for i in range(self.n_outputs):
            # Positive output = pp + nn
            pos_duration = (out_pp[i].get_pulse_width() +
                          out_nn[i].get_pulse_width())
            pos_signal = TimeSignal.from_pulse_width(
                pos_duration, self.phase_duration * 2
            )

            # Negative output = pn + np
            neg_duration = (out_pn[i].get_pulse_width() +
                          out_np[i].get_pulse_width())
            neg_signal = TimeSignal.from_pulse_width(
                neg_duration, self.phase_duration * 2
            )

            output_signals.append(DifferentialTimeSignal(pos_signal, neg_signal))

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

    def process(self, input_signals: List[TimeSignal]) -> Optional[List[TimeSignal]]:
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
                output = self.vmm.compute_single_quadrant(
                    self.pipeline_registers[prev_stage]
                )
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
    def __init__(self, vmm_layers: List[TimeVMM]):
        """
        Initialize chained VMM.

        Args:
            vmm_layers: List of VMM layers to chain
        """
        self.layers = vmm_layers
        self.n_layers = len(vmm_layers)

    def forward(self, input_signals: List[TimeSignal],
                include_intermediate: bool = False) -> List[TimeSignal]:
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