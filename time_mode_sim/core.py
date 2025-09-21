"""
Core classes for time-mode computation simulation framework.
Based on concepts from time-mode analog computation papers.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class SignalLevel(Enum):
    """Digital signal levels for time-encoded signals."""

    LOW = 0
    HIGH = 1


@dataclass
class TimeSignal:
    """
    Represents a time-encoded digital signal.

    Attributes:
        transitions: List of (time, level) tuples representing signal transitions
        duration: Total duration of the signal
        phase_duration: Duration of each phase for multi-phase operations
    """

    transitions: list[tuple[float, SignalLevel]]
    duration: float
    phase_duration: float | None = None

    def __init__(
        self,
        transitions: list[tuple[float, int]] | None = None,
        duration: float = 1.0,
        phase_duration: float | None = None,
    ):
        """Initialize a time signal with transitions."""
        self.duration = duration
        self.phase_duration = phase_duration or duration
        if transitions:
            self.transitions = [(t, SignalLevel(level)) for t, level in transitions]
        else:
            self.transitions = [(0.0, SignalLevel.LOW)]

    def get_level_at(self, time: float) -> SignalLevel:
        """Get signal level at a specific time."""
        if time < 0 or time > self.duration:
            return SignalLevel.LOW

        current_level = SignalLevel.LOW
        for t, level in self.transitions:
            if time >= t:
                current_level = level
            else:
                break
        return current_level

    def get_pulse_width(self, start_time: float = 0.0) -> float:
        """
        Calculate pulse width (time signal is HIGH) from start_time.
        Used for pulse-width encoded values.
        """
        high_time = 0.0
        last_time = start_time
        last_level = self.get_level_at(start_time)

        for t, level in self.transitions:
            if t <= start_time:
                continue
            if t > self.duration:
                break

            if last_level == SignalLevel.HIGH:
                high_time += t - last_time
            last_time = t
            last_level = level

        # Handle final segment
        if last_level == SignalLevel.HIGH and last_time < self.duration:
            high_time += self.duration - last_time

        return high_time

    def get_rising_edge(self) -> float | None:
        """Get time of first LOW to HIGH transition."""
        for i in range(len(self.transitions) - 1):
            if (
                self.transitions[i][1] == SignalLevel.LOW
                and self.transitions[i + 1][1] == SignalLevel.HIGH
            ):
                return self.transitions[i + 1][0]
        return None

    def get_falling_edge(self) -> float | None:
        """Get time of first HIGH to LOW transition."""
        for i in range(len(self.transitions) - 1):
            if (
                self.transitions[i][1] == SignalLevel.HIGH
                and self.transitions[i + 1][1] == SignalLevel.LOW
            ):
                return self.transitions[i + 1][0]
        return None

    @classmethod
    def from_pulse_width(
        cls,
        pulse_width: float,
        duration: float = 1.0,
        start_time: float = 0.0,
        inverted: bool = False,
    ):
        """Create a signal from pulse width encoding."""
        if inverted:
            # For negative values or inverted encoding
            transitions = [(0.0, 1), (start_time, 0), (start_time + pulse_width, 1)]
        else:
            # Standard positive encoding
            transitions = [(0.0, 0), (start_time, 1), (start_time + pulse_width, 0)]
        return cls(transitions, duration)

    @classmethod
    def from_analog_value(
        cls,
        value: float,
        max_value: float = 1.0,
        duration: float = 1.0,
        encoding: str = "rising_edge",
    ):
        """
        Convert analog value to time-encoded signal.

        Args:
            value: Analog value to encode
            max_value: Maximum analog value (for normalization)
            duration: Total signal duration
            encoding: 'rising_edge', 'pulse_width', or 'falling_edge'
        """
        if encoding == "rising_edge":
            # Time of rising edge encodes value (used in some papers)
            edge_time = duration * (1 - value / max_value)
            transitions = [(0.0, 0), (edge_time, 1)]
        elif encoding == "pulse_width":
            # Pulse width encodes value (most common)
            pulse_width = duration * (value / max_value)
            return cls.from_pulse_width(pulse_width, duration)
        elif encoding == "falling_edge":
            # Time of falling edge encodes value
            edge_time = duration * (value / max_value)
            transitions = [(0.0, 1), (edge_time, 0)]
        else:
            raise ValueError(f"Unknown encoding type: {encoding}")

        return cls(transitions, duration)

    def to_analog_value(self, max_value: float = 1.0, encoding: str = "pulse_width") -> float:
        """Convert time-encoded signal back to analog value."""
        if encoding == "pulse_width":
            pulse_width = self.get_pulse_width()
            return (pulse_width / self.duration) * max_value
        elif encoding == "rising_edge":
            edge = self.get_rising_edge()
            if edge is None:
                return 0.0
            return (1 - edge / self.duration) * max_value
        elif encoding == "falling_edge":
            edge = self.get_falling_edge()
            if edge is None:
                return max_value
            return (edge / self.duration) * max_value
        else:
            raise ValueError(f"Unknown encoding type: {encoding}")


class DifferentialTimeSignal:
    """
    Differential time signal for four-quadrant operations.
    Uses two complementary signals to represent signed values.
    """

    def __init__(self, positive: TimeSignal, negative: TimeSignal):
        self.positive = positive
        self.negative = negative
        assert positive.duration == negative.duration, "Signal durations must match"

    @classmethod
    def from_signed_value(cls, value: float, max_value: float = 1.0, duration: float = 1.0):
        """Create differential signal from signed analog value."""
        if value >= 0:
            pos = TimeSignal.from_analog_value(abs(value), max_value, duration)
            neg = TimeSignal.from_analog_value(0, max_value, duration)
        else:
            pos = TimeSignal.from_analog_value(0, max_value, duration)
            neg = TimeSignal.from_analog_value(abs(value), max_value, duration)
        return cls(pos, neg)

    def to_signed_value(self, max_value: float = 1.0) -> float:
        """Convert differential signal to signed analog value."""
        pos_val = self.positive.to_analog_value(max_value)
        neg_val = self.negative.to_analog_value(max_value)
        return pos_val - neg_val


class CurrentSource:
    """
    Represents a programmable current source (e.g., floating-gate transistor).
    Used for weight storage in time-mode VMM.
    """

    def __init__(
        self, current: float, max_current: float = 1e-6, vth: float = 0.5, subthreshold: bool = True
    ):
        """
        Initialize current source.

        Args:
            current: Programmed current value (Amps)
            max_current: Maximum current (for normalization)
            vth: Threshold voltage
            subthreshold: Whether operating in subthreshold regime
        """
        self.current = current
        self.max_current = max_current
        self.vth = vth
        self.subthreshold = subthreshold
        self._noise_sigma = 0.01  # 1% noise by default

    def get_current(self, vgs: float = 1.0, vds: float = 0.5, add_noise: bool = False) -> float:
        """
        Get current with optional DIBL and noise effects.

        Args:
            vgs: Gate-source voltage
            vds: Drain-source voltage
            add_noise: Whether to add random noise
        """
        if self.subthreshold:
            # Simplified subthreshold model with DIBL
            vt = 0.026  # Thermal voltage at room temp
            n = 1.5  # Subthreshold slope factor
            lambda_dibl = 0.1  # DIBL coefficient

            # Basic subthreshold current with DIBL
            i = self.current * np.exp((vgs - self.vth) / (n * vt))
            i *= 1 + lambda_dibl * vds  # DIBL effect

        else:
            # Simple linear model for above threshold
            if vgs > self.vth:
                i = self.current * (vgs - self.vth)
            else:
                i = 0

        if add_noise:
            # Add thermal and 1/f noise
            noise = np.random.normal(0, self._noise_sigma * self.current)
            i += noise

        return max(0, min(i, self.max_current))

    def program(self, target_current: float):
        """Program the current source to a target value."""
        self.current = max(0, min(target_current, self.max_current))
