"""
Basic building blocks for time-mode computation circuits.
Implements mATC, FWPG, time-domain subtraction, and other components.
"""

import numpy as np
from typing import List, Optional, Tuple
from .core import TimeSignal, CurrentSource, SignalLevel


class MonostableMultivibrator:
    """
    Monostable Multivibrator (MSMV) - basic multiplying analog-to-time converter.
    Performs multiplication of analog input (voltage) with weight (current).
    """
    def __init__(self, capacitance: float = 1e-12, vth: float = 0.5):
        """
        Initialize MSMV.

        Args:
            capacitance: Load capacitance (Farads)
            vth: Threshold voltage for output switching
        """
        self.capacitance = capacitance
        self.vth = vth

    def compute(self, input_signal: TimeSignal, current_source: CurrentSource,
                trigger_time: float = 0.0, phase_duration: float = 1.0) -> TimeSignal:
        """
        Compute output pulse from input signal and current source.

        Args:
            input_signal: Time-encoded input signal
            current_source: Weight as current source
            trigger_time: When to trigger the MSMV
            phase_duration: Duration of computation phase

        Returns:
            Time-encoded output signal
        """
        # Get input pulse width (encodes analog value)
        input_duration = input_signal.get_pulse_width()

        # Calculate charge accumulated
        current = current_source.get_current()
        charge = current * input_duration

        # Calculate output pulse width (Q = CV)
        if current > 0:
            output_duration = charge / (current_source.max_current * self.capacitance / self.vth)
        else:
            output_duration = 0

        # Ensure output doesn't exceed phase duration
        output_duration = min(output_duration, phase_duration)

        # Create output pulse starting at trigger_time
        output_signal = TimeSignal.from_pulse_width(
            output_duration,
            duration=phase_duration,
            start_time=trigger_time
        )

        return output_signal


class FixedWidthPulseGenerator:
    """
    Fixed-Width Pulse Generator (FWPG) - maintains signal flow between stages.
    Generates fixed-duration pulse triggered by input edge.
    """
    def __init__(self, pulse_width: float = 0.05, trigger_edge: str = 'falling'):
        """
        Initialize FWPG.

        Args:
            pulse_width: Fixed width of generated pulse
            trigger_edge: 'rising' or 'falling' edge trigger
        """
        self.pulse_width = pulse_width
        self.trigger_edge = trigger_edge

    def generate(self, input_signal: TimeSignal) -> TimeSignal:
        """Generate fixed-width pulse from input signal edge."""
        if self.trigger_edge == 'falling':
            edge_time = input_signal.get_falling_edge()
        else:
            edge_time = input_signal.get_rising_edge()

        if edge_time is None:
            # No edge found, return low signal
            return TimeSignal([(0.0, 0)], input_signal.duration)

        # Generate pulse starting at edge time
        return TimeSignal.from_pulse_width(
            self.pulse_width,
            duration=input_signal.duration,
            start_time=edge_time
        )


class TimeDomainSubtractor:
    """
    Performs time-domain subtraction using XOR/XNOR operations.
    Output = |Input - Delay| with sign information.
    """
    def __init__(self, delay: float):
        """
        Initialize subtractor with fixed delay.

        Args:
            delay: Subtraction delay value
        """
        self.delay = delay

    def subtract(self, input_signal: TimeSignal) -> Tuple[TimeSignal, bool]:
        """
        Perform time-domain subtraction.

        Args:
            input_signal: Input time signal

        Returns:
            Tuple of (result_signal, is_positive)
        """
        # Get input pulse characteristics
        rising_edge = input_signal.get_rising_edge()
        falling_edge = input_signal.get_falling_edge()

        if rising_edge is None or falling_edge is None:
            # Invalid input
            return TimeSignal([(0.0, 0)], input_signal.duration), True

        pulse_width = falling_edge - rising_edge

        # Compute absolute difference
        diff = abs(pulse_width - self.delay)
        is_positive = pulse_width >= self.delay

        # Create output pulse
        output = TimeSignal.from_pulse_width(diff, input_signal.duration)

        return output, is_positive


class TimeToDigitalConverter:
    """
    Converts time-encoded signals to digital values.
    Implements counter-based TDC for simplicity.
    """
    def __init__(self, resolution: int = 8, clock_period: float = 0.001):
        """
        Initialize TDC.

        Args:
            resolution: Number of bits
            clock_period: Clock period for counting
        """
        self.resolution = resolution
        self.clock_period = clock_period
        self.max_count = 2 ** resolution - 1

    def convert(self, time_signal: TimeSignal) -> int:
        """Convert time signal to digital value."""
        pulse_width = time_signal.get_pulse_width()
        counts = int(pulse_width / self.clock_period)
        return min(counts, self.max_count)


class DigitalToTimeConverter:
    """
    Converts digital values to time-encoded signals.
    """
    def __init__(self, resolution: int = 8, time_range: float = 1.0):
        """
        Initialize DTC.

        Args:
            resolution: Number of bits
            time_range: Maximum time duration
        """
        self.resolution = resolution
        self.time_range = time_range
        self.max_value = 2 ** resolution - 1

    def convert(self, digital_value: int) -> TimeSignal:
        """Convert digital value to time signal."""
        # Normalize to [0, 1]
        normalized = min(digital_value, self.max_value) / self.max_value

        # Create pulse with width proportional to value
        pulse_width = normalized * self.time_range
        return TimeSignal.from_pulse_width(pulse_width, self.time_range)


class ChargePump:
    """
    Integrates current over time to accumulate charge.
    Core mechanism for time-domain VMM.
    """
    def __init__(self, capacitance: float = 1e-12, vth: float = 0.5,
                 reset_voltage: float = 0.0):
        """
        Initialize charge pump.

        Args:
            capacitance: Integration capacitor
            vth: Threshold voltage
            reset_voltage: Voltage after reset
        """
        self.capacitance = capacitance
        self.vth = vth
        self.reset_voltage = reset_voltage
        self.voltage = reset_voltage

    def reset(self):
        """Reset capacitor voltage."""
        self.voltage = self.reset_voltage

    def integrate(self, currents: List[float], durations: List[float]) -> float:
        """
        Integrate multiple currents over their respective durations.

        Args:
            currents: List of current values
            durations: List of time durations for each current

        Returns:
            Final voltage on capacitor
        """
        total_charge = sum(i * t for i, t in zip(currents, durations))
        self.voltage += total_charge / self.capacitance
        return self.voltage

    def time_to_threshold(self, constant_current: float) -> float:
        """
        Calculate time to reach threshold with constant current.

        Args:
            constant_current: Constant charging current

        Returns:
            Time to reach threshold
        """
        if constant_current <= 0:
            return float('inf')

        charge_needed = (self.vth - self.voltage) * self.capacitance
        return charge_needed / constant_current


class CompletionDetector:
    """
    Detects completion of computation in asynchronous circuits.
    Can use current sensing or timeout mechanisms.
    """
    def __init__(self, mode: str = 'current', threshold: float = 1e-9,
                 timeout: float = 1.0):
        """
        Initialize completion detector.

        Args:
            mode: 'current' for current sensing, 'timeout' for fixed delay
            threshold: Current threshold for completion
            timeout: Maximum time to wait
        """
        self.mode = mode
        self.threshold = threshold
        self.timeout = timeout

    def detect_completion(self, currents: List[float],
                         start_time: float = 0.0) -> float:
        """
        Detect when computation is complete.

        Args:
            currents: List of current values over time
            start_time: Start time of computation

        Returns:
            Completion time
        """
        if self.mode == 'current':
            # Find when current drops below threshold
            for i, current in enumerate(currents):
                if abs(current) < self.threshold:
                    return start_time + i * (self.timeout / len(currents))
            return start_time + self.timeout
        else:
            # Simple timeout
            return start_time + self.timeout


class SRLatch:
    """
    Set-Reset latch for storing computation results.
    Used in pipelined architectures.
    """
    def __init__(self, initial_state: bool = False):
        """Initialize SR latch."""
        self.state = initial_state
        self.output = TimeSignal([(0.0, int(initial_state))], 1.0)

    def set(self, time: float = 0.0):
        """Set latch output HIGH."""
        self.state = True
        self.output = TimeSignal([(time, 1)], 1.0)

    def reset(self, time: float = 0.0):
        """Reset latch output LOW."""
        self.state = False
        self.output = TimeSignal([(time, 0)], 1.0)

    def update(self, set_signal: TimeSignal, reset_signal: TimeSignal,
               time: float) -> bool:
        """
        Update latch based on input signals.

        Args:
            set_signal: Set input
            reset_signal: Reset input
            time: Current time

        Returns:
            Current latch state
        """
        if set_signal.get_level_at(time) == SignalLevel.HIGH:
            self.set(time)
        elif reset_signal.get_level_at(time) == SignalLevel.HIGH:
            self.reset(time)
        return self.state