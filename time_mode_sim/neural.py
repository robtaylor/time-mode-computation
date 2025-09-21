"""
Neural network components for time-mode computation.
Implements layers, activation functions, and full network architectures.
"""

from dataclasses import dataclass

import numpy as np

from .core import DifferentialTimeSignal, TimeSignal
from .vmm import TimeVMM


class TimeActivation:
    """
    Base class for time-domain activation functions.
    """

    def forward(self, signals: list[TimeSignal]) -> list[TimeSignal]:
        raise NotImplementedError


class SoftminActivation(TimeActivation):
    """
    Softmin activation - fastest neuron wins.
    Common in time-domain networks where earlier signals dominate.
    """

    def __init__(self, inhibition_delay: float = 0.01):
        self.inhibition_delay = inhibition_delay

    def forward(self, signals: list[TimeSignal]) -> list[TimeSignal]:
        """
        Implement winner-take-all based on falling edge timing.
        Earliest falling edge inhibits others.
        """
        # Find earliest falling edge
        min_time = float("inf")
        winner_idx = -1

        for i, signal in enumerate(signals):
            edge_time = signal.get_falling_edge()
            if edge_time is not None and edge_time < min_time:
                min_time = edge_time
                winner_idx = i

        # Create output signals
        outputs = []
        for i in range(len(signals)):
            if i == winner_idx:
                # Winner passes through with slight delay
                outputs.append(signals[i])
            else:
                # Losers are inhibited after winner fires
                inhibited = TimeSignal(
                    [(0.0, 0), (min_time + self.inhibition_delay, 0)], signals[i].duration
                )
                outputs.append(inhibited)

        return outputs


class ReLUActivation(TimeActivation):
    """
    ReLU activation using AND gate with bias signal.
    Passes positive values, blocks negative.
    """

    def __init__(self, threshold_time: float = 0.1):
        self.threshold_time = threshold_time

    def forward(self, signals: list[TimeSignal]) -> list[TimeSignal]:
        """
        Apply ReLU by comparing pulse width to threshold.
        """
        outputs = []
        for signal in signals:
            pulse_width = signal.get_pulse_width()

            if pulse_width > self.threshold_time:
                # Positive - pass through (no adjustment to maintain signal strength)
                output = signal
            else:
                # Negative - output zero
                output = TimeSignal([(0.0, 0)], signal.duration)

            outputs.append(output)

        return outputs


class ThresholdActivation(TimeActivation):
    """
    Simple threshold activation - binary output.
    """

    def __init__(self, threshold: float = 0.5, output_width: float = 0.8):
        self.threshold = threshold
        self.output_width = output_width

    def forward(self, signals: list[TimeSignal]) -> list[TimeSignal]:
        outputs = []
        for signal in signals:
            pulse_width = signal.get_pulse_width()

            if pulse_width > self.threshold:
                # Fire
                output = TimeSignal.from_pulse_width(self.output_width, signal.duration)
            else:
                # Don't fire
                output = TimeSignal([(0.0, 0)], signal.duration)

            outputs.append(output)

        return outputs


@dataclass
class TimeNeuralLayer:
    """
    Single neural network layer in time domain.
    """

    input_size: int
    output_size: int
    weights: np.ndarray | None = None
    activation: TimeActivation | None = None
    vmm_params: dict = None

    def __post_init__(self):
        if self.weights is None:
            # Initialize with random weights
            self.weights = np.random.randn(self.output_size, self.input_size) * 0.1

        if self.vmm_params is None:
            self.vmm_params = {
                "max_current": 1e-6,
                "capacitance": 1e-12,
                "vth": 0.5,
                "phase_duration": 1.0,
            }

        # Create VMM for this layer
        self.vmm = TimeVMM(self.weights, **self.vmm_params)

    def forward(self, inputs: list[TimeSignal]) -> list[TimeSignal]:
        """
        Forward pass through layer.
        """
        # VMM computation
        outputs = self.vmm.compute_single_quadrant(inputs)

        # Apply activation if present
        if self.activation is not None:
            outputs = self.activation.forward(outputs)

        return outputs

    def forward_differential(
        self, inputs: list[DifferentialTimeSignal]
    ) -> list[DifferentialTimeSignal]:
        """
        Forward pass with differential signals for signed weights.
        """
        outputs = self.vmm.compute_four_quadrant(inputs)

        # Note: Most activations need modification for differential signals
        # This is left as future work

        return outputs


class TimeNeuralNetwork:
    """
    Multi-layer time-domain neural network.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activations: list[TimeActivation] | None = None,
        vmm_params: dict | None = None,
    ):
        """
        Initialize network.

        Args:
            layer_sizes: List of layer sizes [input, hidden1, ..., output]
            activations: Activation functions for each layer (except input)
            vmm_params: Parameters for VMM modules
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1

        if activations is None:
            # Default to ReLU for hidden, softmin for output
            activations = [ReLUActivation() for _ in range(self.n_layers - 1)]
            activations.append(SoftminActivation())

        self.layers = []
        for i in range(self.n_layers):
            layer = TimeNeuralLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activations[i] if i < len(activations) else None,
                vmm_params=vmm_params,
            )
            self.layers.append(layer)

    def forward(self, inputs: list[TimeSignal]) -> list[TimeSignal]:
        """
        Forward pass through entire network.
        """
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def set_weights(self, weights: list[np.ndarray]):
        """
        Set weights for all layers.
        """
        for layer, w in zip(self.layers, weights, strict=False):
            layer.weights = w
            layer.vmm = TimeVMM(w, **layer.vmm_params)

    def get_weights(self) -> list[np.ndarray]:
        """
        Get weights from all layers.
        """
        return [layer.weights for layer in self.layers]


class ConvolutionalTimeLayer:
    """
    Time-domain convolutional layer.
    Implements convolution through time-multiplexed VMM operations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        vmm_params: dict | None = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Initialize kernels
        self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1

        if vmm_params is None:
            vmm_params = {
                "max_current": 1e-6,
                "capacitance": 1e-12,
                "vth": 0.5,
                "phase_duration": 1.0,
            }
        self.vmm_params = vmm_params

    def im2col(self, input_map: np.ndarray) -> np.ndarray:
        """
        Convert input feature map to column matrix for VMM.
        """
        h, w = input_map.shape[1:3]
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1

        col = np.zeros((self.kernel_size * self.kernel_size * self.in_channels, out_h * out_w))

        idx = 0
        for i in range(0, h - self.kernel_size + 1, self.stride):
            for j in range(0, w - self.kernel_size + 1, self.stride):
                patch = input_map[:, i : i + self.kernel_size, j : j + self.kernel_size]
                col[:, idx] = patch.flatten()
                idx += 1

        return col

    def forward(self, input_signals: list[list[TimeSignal]]) -> list[list[TimeSignal]]:
        """
        Forward pass through convolutional layer.
        Input is 2D array of time signals representing feature map.
        """
        # This is a simplified implementation
        # Full implementation would require careful time-multiplexing
        raise NotImplementedError("Convolutional layers require time-multiplexing")


class RecurrentTimeLayer:
    """
    Recurrent layer for time-domain processing.
    Processes sequential inputs with memory.
    """

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, vmm_params: dict | None = None
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weight matrices
        self.W_ih = np.random.randn(hidden_size, input_size) * 0.1
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W_ho = np.random.randn(output_size, hidden_size) * 0.1

        if vmm_params is None:
            vmm_params = {
                "max_current": 1e-6,
                "capacitance": 1e-12,
                "vth": 0.5,
                "phase_duration": 1.0,
            }

        # Create VMMs
        self.vmm_ih = TimeVMM(self.W_ih, **vmm_params)
        self.vmm_hh = TimeVMM(self.W_hh, **vmm_params)
        self.vmm_ho = TimeVMM(self.W_ho, **vmm_params)

        # Hidden state
        self.hidden_state = None

    def reset_hidden(self):
        """
        Reset hidden state.
        """
        self.hidden_state = [TimeSignal([(0.0, 0)], 1.0) for _ in range(self.hidden_size)]

    def forward(self, input_sequence: list[list[TimeSignal]]) -> list[list[TimeSignal]]:
        """
        Process sequence of inputs.
        """
        if self.hidden_state is None:
            self.reset_hidden()

        outputs = []

        for inputs in input_sequence:
            # Input to hidden
            h_from_input = self.vmm_ih.compute_single_quadrant(inputs)

            # Hidden to hidden
            h_from_hidden = self.vmm_hh.compute_single_quadrant(self.hidden_state)

            # Combine (simplified - real implementation would need careful timing)
            new_hidden = []
            for h_i, h_h in zip(h_from_input, h_from_hidden, strict=False):
                # Add pulse widths (simplified combination)
                combined_width = h_i.get_pulse_width() + h_h.get_pulse_width()
                combined_width = min(combined_width, h_i.duration * 0.9)
                new_hidden.append(TimeSignal.from_pulse_width(combined_width, h_i.duration))

            self.hidden_state = new_hidden

            # Hidden to output
            output = self.vmm_ho.compute_single_quadrant(self.hidden_state)
            outputs.append(output)

        return outputs


class TimeAutoencoder:
    """
    Autoencoder architecture in time domain.
    """

    def __init__(self, input_size: int, encoding_size: int, hidden_sizes: list[int] | None = None):
        """
        Initialize autoencoder.

        Args:
            input_size: Size of input
            encoding_size: Size of encoding (bottleneck)
            hidden_sizes: Sizes of hidden layers
        """
        if hidden_sizes is None:
            hidden_sizes = [input_size // 2]

        # Build encoder
        encoder_sizes = [input_size] + hidden_sizes + [encoding_size]
        self.encoder = TimeNeuralNetwork(
            encoder_sizes, activations=[ReLUActivation() for _ in range(len(encoder_sizes) - 1)]
        )

        # Build decoder (mirror of encoder)
        decoder_sizes = [encoding_size] + hidden_sizes[::-1] + [input_size]
        self.decoder = TimeNeuralNetwork(
            decoder_sizes,
            activations=[ReLUActivation() for _ in range(len(decoder_sizes) - 2)]
            + [ThresholdActivation()],
        )

    def encode(self, inputs: list[TimeSignal]) -> list[TimeSignal]:
        """
        Encode inputs to latent representation.
        """
        return self.encoder.forward(inputs)

    def decode(self, encoding: list[TimeSignal]) -> list[TimeSignal]:
        """
        Decode from latent representation.
        """
        return self.decoder.forward(encoding)

    def forward(self, inputs: list[TimeSignal]) -> list[TimeSignal]:
        """
        Full forward pass (encode then decode).
        """
        encoding = self.encode(inputs)
        return self.decode(encoding)
