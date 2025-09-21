"""
Visualization and analysis tools for time-mode computation.
Provides plotting, waveform analysis, and performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict
from .core import TimeSignal, DifferentialTimeSignal


class SignalVisualizer:
    """
    Visualize time-encoded signals.
    """
    def __init__(self, figsize: Tuple[float, float] = (10, 6)):
        self.figsize = figsize
    
    def plot_signal(self, signal: TimeSignal, ax: Optional[plt.Axes] = None,
                   label: str = "Signal", color: str = "blue",
                   show_edges: bool = True) -> plt.Axes:
        """
        Plot a single time signal.
        
        Args:
            signal: Time signal to plot
            ax: Matplotlib axes (creates new if None)
            label: Label for the signal
            color: Color for the plot
            show_edges: Whether to mark edges
        
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create time points for plotting
        times = [0]
        levels = [signal.get_level_at(0).value]
        
        for t, level in signal.transitions:
            # Add point just before transition
            times.append(t - 1e-9)
            levels.append(levels[-1])
            # Add transition
            times.append(t)
            levels.append(level.value)
        
        # Add final point
        times.append(signal.duration)
        levels.append(levels[-1])
        
        # Plot signal
        ax.plot(times, levels, color=color, linewidth=2, label=label)
        
        # Mark edges if requested
        if show_edges:
            rising = signal.get_rising_edge()
            falling = signal.get_falling_edge()
            
            if rising is not None:
                ax.axvline(rising, color='green', linestyle='--', alpha=0.5)
                ax.text(rising, 0.5, f'R:{rising:.3f}', rotation=90)
            
            if falling is not None:
                ax.axvline(falling, color='red', linestyle='--', alpha=0.5)
                ax.text(falling, 0.5, f'F:{falling:.3f}', rotation=90)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Signal Level')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax
    
    def plot_multiple_signals(self, signals: List[TimeSignal],
                            labels: Optional[List[str]] = None,
                            title: str = "Time Signals") -> plt.Figure:
        """
        Plot multiple signals on separate subplots.
        """
        n_signals = len(signals)
        fig, axes = plt.subplots(n_signals, 1, figsize=(self.figsize[0], 
                                                        self.figsize[1] * n_signals / 3),
                                sharex=True)
        
        if n_signals == 1:
            axes = [axes]
        
        if labels is None:
            labels = [f"Signal {i}" for i in range(n_signals)]
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_signals))
        
        for i, (signal, label, color) in enumerate(zip(signals, labels, colors)):
            self.plot_signal(signal, axes[i], label, color[:3])
            axes[i].set_xlabel('')  # Remove x-label except for bottom
        
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    def plot_differential_signal(self, signal: DifferentialTimeSignal,
                                ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot differential time signal.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot positive and negative components
        self.plot_signal(signal.positive, ax, "Positive", "blue", False)
        self.plot_signal(signal.negative, ax, "Negative", "red", False)
        
        # Add difference visualization
        times = np.linspace(0, signal.positive.duration, 1000)
        diff = []
        for t in times:
            pos = signal.positive.get_level_at(t).value
            neg = signal.negative.get_level_at(t).value
            diff.append(pos - neg)
        
        ax2 = ax.twinx()
        ax2.plot(times, diff, 'g--', alpha=0.5, label='Difference')
        ax2.set_ylabel('Difference', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        
        ax.set_title('Differential Signal')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        return ax


class WaveformAnalyzer:
    """
    Analyze time-domain waveforms.
    """
    
    @staticmethod
    def compute_pulse_statistics(signals: List[TimeSignal]) -> Dict:
        """
        Compute statistics of pulse widths.
        """
        widths = [s.get_pulse_width() for s in signals]
        
        return {
            'mean': np.mean(widths),
            'std': np.std(widths),
            'min': np.min(widths),
            'max': np.max(widths),
            'median': np.median(widths)
        }
    
    @staticmethod
    def compute_edge_statistics(signals: List[TimeSignal]) -> Dict:
        """
        Compute statistics of edge timings.
        """
        rising_edges = [s.get_rising_edge() for s in signals]
        falling_edges = [s.get_falling_edge() for s in signals]
        
        # Filter None values
        rising_edges = [e for e in rising_edges if e is not None]
        falling_edges = [e for e in falling_edges if e is not None]
        
        stats = {}
        
        if rising_edges:
            stats['rising'] = {
                'mean': np.mean(rising_edges),
                'std': np.std(rising_edges),
                'min': np.min(rising_edges),
                'max': np.max(rising_edges)
            }
        
        if falling_edges:
            stats['falling'] = {
                'mean': np.mean(falling_edges),
                'std': np.std(falling_edges),
                'min': np.min(falling_edges),
                'max': np.max(falling_edges)
            }
        
        return stats
    
    @staticmethod
    def compute_timing_jitter(signals: List[TimeSignal],
                            reference_signal: Optional[TimeSignal] = None) -> float:
        """
        Compute timing jitter relative to reference.
        """
        if reference_signal is None:
            # Use mean as reference
            ref_width = np.mean([s.get_pulse_width() for s in signals])
        else:
            ref_width = reference_signal.get_pulse_width()
        
        deviations = [abs(s.get_pulse_width() - ref_width) for s in signals]
        return np.std(deviations)
    
    @staticmethod
    def compute_snr(signal: TimeSignal, noise_signals: List[TimeSignal]) -> float:
        """
        Compute signal-to-noise ratio.
        """
        signal_power = signal.get_pulse_width() ** 2
        
        noise_powers = [(s.get_pulse_width() - signal.get_pulse_width()) ** 2 
                       for s in noise_signals]
        noise_power = np.mean(noise_powers)
        
        if noise_power == 0:
            return float('inf')
        
        return 10 * np.log10(signal_power / noise_power)


class PerformanceAnalyzer:
    """
    Analyze performance metrics of time-mode circuits.
    """
    
    def __init__(self, capacitance: float = 1e-12, vdd: float = 1.0):
        self.capacitance = capacitance
        self.vdd = vdd
    
    def compute_energy_per_operation(self, n_transitions: int) -> float:
        """
        Compute energy per operation.
        E = n * C * V^2
        """
        return n_transitions * self.capacitance * self.vdd ** 2
    
    def compute_throughput(self, signals: List[TimeSignal],
                          phase_duration: float) -> float:
        """
        Compute operations per second.
        """
        n_ops = len(signals)
        total_time = phase_duration
        return n_ops / total_time
    
    def compute_efficiency(self, n_operations: int, total_energy: float) -> float:
        """
        Compute operations per Joule.
        """
        if total_energy == 0:
            return float('inf')
        return n_operations / total_energy
    
    def analyze_vmm_performance(self, input_size: int, output_size: int,
                               phase_duration: float) -> Dict:
        """
        Analyze VMM performance metrics.
        """
        # Number of multiply-accumulate operations
        n_macs = input_size * output_size
        
        # Energy per MAC (simplified model)
        # Assumes 2 transitions per MAC
        energy_per_mac = self.compute_energy_per_operation(2)
        total_energy = n_macs * energy_per_mac
        
        # Throughput
        throughput = n_macs / phase_duration
        
        # Energy efficiency
        efficiency = n_macs / total_energy
        
        return {
            'n_macs': n_macs,
            'energy_per_mac': energy_per_mac,
            'total_energy': total_energy,
            'throughput': throughput,
            'efficiency': efficiency,
            'fj_per_op': energy_per_mac * 1e15  # Convert to fJ
        }


class NetworkVisualizer:
    """
    Visualize neural network architectures and activations.
    """
    
    def __init__(self, figsize: Tuple[float, float] = (12, 8)):
        self.figsize = figsize
    
    def plot_weight_matrix(self, weights: np.ndarray, title: str = "Weights",
                          cmap: str = 'RdBu_r') -> plt.Figure:
        """
        Visualize weight matrix.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        im = ax.imshow(weights, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax)
        
        ax.set_xlabel('Input Neurons')
        ax.set_ylabel('Output Neurons')
        ax.set_title(title)
        
        return fig
    
    def plot_layer_activations(self, layer_outputs: List[List[TimeSignal]],
                              layer_names: Optional[List[str]] = None) -> plt.Figure:
        """
        Plot activations for each layer.
        """
        n_layers = len(layer_outputs)
        
        if layer_names is None:
            layer_names = [f"Layer {i}" for i in range(n_layers)]
        
        fig, axes = plt.subplots(1, n_layers, figsize=(self.figsize[0] * n_layers / 3,
                                                       self.figsize[1]))
        
        if n_layers == 1:
            axes = [axes]
        
        for i, (outputs, name) in enumerate(zip(layer_outputs, layer_names)):
            # Convert to pulse widths for visualization
            widths = [s.get_pulse_width() for s in outputs]
            
            axes[i].bar(range(len(widths)), widths)
            axes[i].set_xlabel('Neuron Index')
            axes[i].set_ylabel('Activation (Pulse Width)')
            axes[i].set_title(name)
        
        plt.tight_layout()
        return fig
    
    def plot_training_history(self, history: Dict, metrics: List[str] = None) -> plt.Figure:
        """
        Plot training history (loss, accuracy, etc.).
        """
        if metrics is None:
            metrics = list(history.keys())
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(self.figsize[0] * n_metrics / 2,
                                                        self.figsize[1] / 2))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in history:
                axes[i].plot(history[metric])
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric.capitalize())
                axes[i].set_title(f'Training {metric}')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class TimingDiagram:
    """
    Create timing diagrams for multi-phase operations.
    """
    
    def __init__(self, figsize: Tuple[float, float] = (14, 8)):
        self.figsize = figsize
    
    def create_diagram(self, phases: List[Dict], title: str = "Timing Diagram") -> plt.Figure:
        """
        Create timing diagram showing multiple phases.
        
        Args:
            phases: List of phase dictionaries with 'name', 'start', 'duration', 'signals'
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        y_pos = 0
        colors = plt.cm.Set3(np.linspace(0, 1, len(phases)))
        
        for phase, color in zip(phases, colors):
            # Draw phase block
            rect = plt.Rectangle((phase['start'], y_pos), phase['duration'], 0.8,
                                facecolor=color, edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
            
            # Add phase label
            ax.text(phase['start'] + phase['duration']/2, y_pos + 0.4,
                   phase['name'], ha='center', va='center')
            
            # Draw signals if provided
            if 'signals' in phase:
                for i, signal in enumerate(phase['signals']):
                    signal_y = y_pos - 0.2 * (i + 1)
                    times = np.linspace(phase['start'], 
                                      phase['start'] + phase['duration'], 100)
                    levels = [signal.get_level_at(t - phase['start']).value * 0.15 
                             for t in times]
                    ax.plot(times, np.array(levels) + signal_y, 'b-', linewidth=1)
            
            y_pos += 1
        
        ax.set_xlim(0, max(p['start'] + p['duration'] for p in phases))
        ax.set_ylim(-0.5, len(phases))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Phases')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return fig