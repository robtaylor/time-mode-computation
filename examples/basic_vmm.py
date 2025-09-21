"""
Basic VMM example demonstrating time-mode vector-matrix multiplication.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to import time_mode_sim
sys.path.insert(0, str(Path(__file__).parent.parent))

from time_mode_sim import (
    DifferentialTimeSignal,
    PerformanceAnalyzer,
    SignalVisualizer,
    TimeSignal,
    TimeVMM,
)


def example_single_quadrant_vmm():
    """
    Demonstrate single-quadrant VMM operation.
    """
    print("Single-Quadrant VMM Example")
    print("=" * 40)

    # Define weight matrix (3x4)
    weights = np.array([[0.8, 0.2, 0.5, 0.1], [0.3, 0.7, 0.4, 0.6], [0.1, 0.9, 0.2, 0.8]])

    print(f"Weight matrix shape: {weights.shape}")
    print(f"Weights:\n{weights}")

    # Create VMM module
    vmm = TimeVMM(weights=weights, max_current=1e-6, capacitance=1e-12, vth=0.5, phase_duration=1.0)

    # Create input vector as time signals
    input_values = [0.7, 0.3, 0.5, 0.9]
    input_signals = [
        TimeSignal.from_analog_value(val, encoding="pulse_width") for val in input_values
    ]

    print(f"\nInput vector: {input_values}")

    # Compute VMM
    output_signals = vmm.compute_single_quadrant(input_signals)

    # Convert outputs back to analog
    output_values = [sig.to_analog_value(encoding="pulse_width") for sig in output_signals]

    print(f"Output vector: {output_values}")

    # Verify against numpy
    expected = weights @ np.array(input_values)
    print(f"Expected (numpy): {expected}")
    print(f"Max error: {np.max(np.abs(output_values - expected)):.6f}")

    # Visualize signals
    viz = SignalVisualizer()

    # Plot input signals
    fig_input = viz.plot_multiple_signals(
        input_signals,
        labels=[f"Input {i}" for i in range(len(input_signals))],
        title="Input Signals",
    )

    # Plot output signals
    fig_output = viz.plot_multiple_signals(
        output_signals,
        labels=[f"Output {i}" for i in range(len(output_signals))],
        title="Output Signals",
    )

    # Performance analysis
    perf = PerformanceAnalyzer()
    metrics = perf.analyze_vmm_performance(
        input_size=len(input_signals),
        output_size=len(output_signals),
        phase_duration=vmm.phase_duration,
    )

    print("\nPerformance Metrics:")
    print(f"  MACs: {metrics['n_macs']}")
    print(f"  Energy/MAC: {metrics['fj_per_op']:.2f} fJ")
    print(f"  Throughput: {metrics['throughput']:.2e} ops/s")
    print(f"  Efficiency: {metrics['efficiency']:.2e} ops/J")

    return fig_input, fig_output


def example_four_quadrant_vmm():
    """
    Demonstrate four-quadrant VMM with signed weights and inputs.
    """
    print("\n" + "=" * 40)
    print("Four-Quadrant VMM Example")
    print("=" * 40)

    # Define weight matrix with positive and negative values
    weights = np.array([[0.5, -0.3, 0.8], [-0.2, 0.7, -0.4], [0.9, -0.6, 0.1]])

    print(f"Weight matrix:\n{weights}")

    # Create VMM module
    vmm = TimeVMM(weights=weights, max_current=1e-6, capacitance=1e-12, vth=0.5, phase_duration=1.0)

    # Create differential input signals
    input_values = [0.5, -0.7, 0.3]
    input_signals = [DifferentialTimeSignal.from_signed_value(val) for val in input_values]

    print(f"\nInput vector: {input_values}")

    # Compute four-quadrant VMM
    output_signals = vmm.compute_four_quadrant(input_signals)

    # Convert outputs back to signed values
    output_values = [sig.to_signed_value() for sig in output_signals]

    print(f"Output vector: {output_values}")

    # Verify against numpy
    expected = weights @ np.array(input_values)
    print(f"Expected (numpy): {expected}")
    print(f"Max error: {np.max(np.abs(output_values - expected)):.6f}")

    # Visualize differential signals
    viz = SignalVisualizer()
    fig, axes = plt.subplots(len(output_signals), 1, figsize=(10, 8))

    if len(output_signals) == 1:
        axes = [axes]

    for i, (sig, ax) in enumerate(zip(output_signals, axes, strict=False)):
        viz.plot_differential_signal(sig, ax)
        ax.set_title(f"Output {i}: {output_values[i]:.3f}")

    plt.tight_layout()

    return fig


def example_pipelined_vmm():
    """
    Demonstrate pipelined VMM for higher throughput.
    """
    print("\n" + "=" * 40)
    print("Pipelined VMM Example")
    print("=" * 40)

    from time_mode_sim import PipelinedVMM

    # Create base VMM
    weights = np.random.randn(2, 3) * 0.5
    base_vmm = TimeVMM(weights)

    # Create pipelined version
    pipelined = PipelinedVMM(base_vmm, pipeline_depth=2)

    print(f"Pipeline depth: {pipelined.pipeline_depth}")
    print(f"Weight matrix shape: {weights.shape}")

    # Process multiple input batches
    results = []
    for batch in range(4):
        input_values = np.random.rand(3)
        inputs = [TimeSignal.from_analog_value(val, encoding="pulse_width") for val in input_values]

        output = pipelined.process(inputs)

        if output is not None:
            output_values = [sig.to_analog_value(encoding="pulse_width") for sig in output]
            results.append(output_values)
            print(
                f"Batch {batch}: Input={input_values.round(3)}, "
                f"Output={np.array(output_values).round(3)}"
            )
        else:
            print(f"Batch {batch}: Pipeline filling...")

    print(f"\nTotal outputs produced: {len(results)}")


if __name__ == "__main__":
    import os

    # Run examples
    fig1, fig2 = example_single_quadrant_vmm()
    fig3 = example_four_quadrant_vmm()
    example_pipelined_vmm()

    # Show plots only if not in headless mode
    if os.environ.get("DISPLAY") or os.environ.get("MPLBACKEND") != "Agg":
        try:
            plt.show(block=False)
            plt.pause(0.1)
        except Exception:
            print("\nCannot display plots in current environment")
            print("Saving plots to files instead...")
            fig1.savefig("vmm_inputs.png")
            fig2.savefig("vmm_outputs.png")
            fig3.savefig("vmm_differential.png")
            print("Plots saved.")
