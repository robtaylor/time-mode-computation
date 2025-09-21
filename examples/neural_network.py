"""
Neural network example using time-mode computation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from time_mode_sim import (
    TimeSignal,
    TimeNeuralNetwork,
    SoftminActivation,
    ReLUActivation,
    NetworkVisualizer,
    PerformanceAnalyzer
)


def create_xor_dataset():
    """
    Create XOR dataset for testing.
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    return X, y


def example_simple_network():
    """
    Demonstrate a simple feedforward network.
    """
    print("Simple Neural Network Example")
    print("="*40)
    
    # Create network architecture
    layer_sizes = [2, 4, 1]  # 2 inputs, 4 hidden, 1 output
    
    # Create network with custom activations
    activations = [
        ReLUActivation(threshold_time=0.3),  # Hidden layer
        SoftminActivation()  # Output layer
    ]
    
    network = TimeNeuralNetwork(
        layer_sizes=layer_sizes,
        activations=activations,
        vmm_params={'phase_duration': 1.0}
    )
    
    print(f"Network architecture: {layer_sizes}")
    print(f"Number of layers: {network.n_layers}")
    
    # Create XOR dataset
    X, y = create_xor_dataset()
    
    # Set trained weights (pre-computed for XOR)
    # These would normally come from training
    weights = [
        np.array([[2.5, 2.5], [-2.5, -2.5], [2.5, -2.5], [-2.5, 2.5]]),  # Hidden
        np.array([[1.0], [1.0], [-1.0], [-1.0]])  # Output
    ]
    network.set_weights(weights)
    
    print("\nProcessing XOR inputs:")
    print("-" * 30)
    
    # Process each input
    outputs = []
    for i, x in enumerate(X):
        # Convert to time signals
        input_signals = [
            TimeSignal.from_analog_value(val, encoding='pulse_width')
            for val in x
        ]
        
        # Forward pass
        output_signals = network.forward(input_signals)
        
        # Convert output back to analog
        output_value = output_signals[0].to_analog_value(encoding='pulse_width')
        outputs.append(output_value)
        
        print(f"Input: {x} -> Output: {output_value:.3f} (Expected: {y[i][0]})")
    
    # Calculate accuracy
    outputs = np.array(outputs)
    predictions = (outputs > 0.5).astype(int)
    accuracy = np.mean(predictions == y.flatten())
    print(f"\nAccuracy: {accuracy * 100:.1f}%")
    
    return network, X, outputs


def example_multilayer_network():
    """
    Demonstrate a deeper network with multiple hidden layers.
    """
    print("\n" + "="*40)
    print("Multi-Layer Network Example")
    print("="*40)
    
    # Create deeper architecture
    layer_sizes = [4, 8, 6, 3]  # 4 inputs, 2 hidden layers, 3 outputs
    
    network = TimeNeuralNetwork(layer_sizes=layer_sizes)
    
    print(f"Network architecture: {layer_sizes}")
    print(f"Total parameters: {sum(l1*l2 for l1, l2 in zip(layer_sizes[:-1], layer_sizes[1:]))}")
    
    # Create random input
    input_values = np.random.rand(4)
    input_signals = [
        TimeSignal.from_analog_value(val, encoding='pulse_width')
        for val in input_values
    ]
    
    print(f"\nInput: {input_values.round(3)}")
    
    # Forward pass and collect intermediate outputs
    layer_outputs = []
    current = input_signals
    
    for i, layer in enumerate(network.layers):
        current = layer.forward(current)
        layer_outputs.append(current)
        
        # Convert to analog for display
        values = [s.to_analog_value() for s in current]
        print(f"Layer {i+1} output: {np.array(values).round(3)}")
    
    # Visualize network
    viz = NetworkVisualizer()
    
    # Plot weight matrices
    fig_weights, axes = plt.subplots(1, len(network.layers), figsize=(12, 4))
    for i, (layer, ax) in enumerate(zip(network.layers, axes)):
        im = ax.imshow(layer.weights, cmap='RdBu_r', aspect='auto')
        ax.set_title(f"Layer {i+1} Weights")
        ax.set_xlabel('Inputs')
        ax.set_ylabel('Outputs')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    # Plot activations
    fig_activations = viz.plot_layer_activations(
        layer_outputs,
        layer_names=[f"Layer {i+1}" for i in range(len(layer_outputs))]
    )
    
    return network, fig_weights, fig_activations


def example_performance_analysis():
    """
    Analyze performance metrics of time-mode neural networks.
    """
    print("\n" + "="*40)
    print("Performance Analysis")
    print("="*40)
    
    # Different network sizes for comparison
    architectures = [
        [10, 20, 10],
        [100, 200, 100],
        [784, 256, 128, 10]  # MNIST-like
    ]
    
    perf_analyzer = PerformanceAnalyzer(
        capacitance=1e-12,  # 1 fF
        vdd=0.5  # 0.5V operation
    )
    
    print("\nNetwork Performance Comparison:")
    print("-" * 60)
    print(f"{'Architecture':<20} {'MACs':<10} {'Energy (pJ)':<12} {'fJ/Op':<10}")
    print("-" * 60)
    
    results = []
    
    for arch in architectures:
        total_macs = 0
        total_energy = 0
        
        # Calculate for each layer
        for i in range(len(arch) - 1):
            metrics = perf_analyzer.analyze_vmm_performance(
                input_size=arch[i],
                output_size=arch[i+1],
                phase_duration=1e-6  # 1 μs
            )
            total_macs += metrics['n_macs']
            total_energy += metrics['total_energy']
        
        fj_per_op = (total_energy / total_macs) * 1e15 if total_macs > 0 else 0
        
        arch_str = "×".join(map(str, arch))
        print(f"{arch_str:<20} {total_macs:<10} {total_energy*1e12:<12.3f} {fj_per_op:<10.2f}")
        
        results.append({
            'architecture': arch,
            'macs': total_macs,
            'energy': total_energy,
            'fj_per_op': fj_per_op
        })
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # MACs vs Energy
    macs = [r['macs'] for r in results]
    energies = [r['energy'] * 1e12 for r in results]  # Convert to pJ
    
    ax1.scatter(macs, energies, s=100)
    for i, arch in enumerate(architectures):
        ax1.annotate("×".join(map(str, arch)), 
                    (macs[i], energies[i]),
                    xytext=(5, 5), textcoords='offset points')
    
    ax1.set_xlabel('Total MACs')
    ax1.set_ylabel('Total Energy (pJ)')
    ax1.set_title('Energy vs Computation')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Energy efficiency
    sizes = [sum(a[:-1]) for a in architectures]  # Total input neurons
    fj_ops = [r['fj_per_op'] for r in results]
    
    ax2.bar(range(len(architectures)), fj_ops)
    ax2.set_xticks(range(len(architectures)))
    ax2.set_xticklabels(["×".join(map(str, a)) for a in architectures], rotation=45)
    ax2.set_ylabel('Energy Efficiency (fJ/Op)')
    ax2.set_title('Energy Efficiency Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Compare to state-of-the-art
    print("\n" + "="*40)
    print("Comparison to Literature:")
    print("-" * 40)
    print("This work (simulated): ~{:.1f} fJ/Op".format(np.mean(fj_ops)))
    print("Bavandpour et al. (2019): 7 fJ/Op @ 55nm")
    print("Digital CMOS (45nm): ~100-1000 fJ/Op")
    print("Analog current-mode: ~10-100 fJ/Op")
    
    return fig


def example_autoencoder():
    """
    Demonstrate autoencoder for dimensionality reduction.
    """
    print("\n" + "="*40)
    print("Autoencoder Example")
    print("="*40)
    
    from time_mode_sim import TimeAutoencoder
    
    # Create autoencoder
    input_size = 8
    encoding_size = 2
    
    autoencoder = TimeAutoencoder(
        input_size=input_size,
        encoding_size=encoding_size,
        hidden_sizes=[4]
    )
    
    print(f"Autoencoder architecture:")
    print(f"  Encoder: {input_size} -> 4 -> {encoding_size}")
    print(f"  Decoder: {encoding_size} -> 4 -> {input_size}")
    
    # Create test input
    input_values = np.random.rand(input_size)
    input_signals = [
        TimeSignal.from_analog_value(val, encoding='pulse_width')
        for val in input_values
    ]
    
    print(f"\nOriginal input: {input_values.round(3)}")
    
    # Encode
    encoded = autoencoder.encode(input_signals)
    encoded_values = [s.to_analog_value() for s in encoded]
    print(f"Encoded (dim={encoding_size}): {np.array(encoded_values).round(3)}")
    
    # Decode
    reconstructed = autoencoder.decode(encoded)
    reconstructed_values = [s.to_analog_value() for s in reconstructed]
    print(f"Reconstructed: {np.array(reconstructed_values).round(3)}")
    
    # Calculate reconstruction error
    error = np.mean((input_values - reconstructed_values)**2)
    print(f"\nMSE Reconstruction Error: {error:.6f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original
    axes[0].bar(range(input_size), input_values)
    axes[0].set_title('Original Input')
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Value')
    
    # Encoded
    axes[1].bar(range(encoding_size), encoded_values)
    axes[1].set_title(f'Encoded (dim={encoding_size})')
    axes[1].set_xlabel('Dimension')
    
    # Reconstructed
    axes[2].bar(range(input_size), reconstructed_values, alpha=0.7, label='Reconstructed')
    axes[2].bar(range(input_size), input_values, alpha=0.5, label='Original')
    axes[2].set_title('Reconstruction Comparison')
    axes[2].set_xlabel('Dimension')
    axes[2].legend()
    
    plt.tight_layout()
    
    return autoencoder, fig


if __name__ == "__main__":
    # Run examples
    network1, X, outputs = example_simple_network()
    network2, fig_w, fig_a = example_multilayer_network()
    fig_perf = example_performance_analysis()
    autoencoder, fig_ae = example_autoencoder()
    
    # Show all plots
    plt.show()