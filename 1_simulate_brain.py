import numpy as np
import os


def main():
    print("Initializing Neuralink 'Telepathy' Simulator...")

    # Simulate a 96-channel array (Standard BCI implant size)
    num_neurons = 96
    num_timesteps = 25000
    dt = 0.05  # 50ms time bins (standard for real-time decoding)

    # 1. BIOLOGICAL ENCODING: Assign each neuron a 'preferred direction'
    np.random.seed(42)
    preferred_directions = np.random.uniform(0, 2 * np.pi, num_neurons)
    baseline_rates = np.random.uniform(5, 20, num_neurons)  # Spikes per second
    modulation_depths = np.random.uniform(10, 30, num_neurons)

    # 2. INTENT: Generate smooth 2D cursor movements (Human thinking about moving a mouse)
    print("Generating intended cursor trajectories...")

    # We use a moving average to create smooth, realistic hand-like trajectories
    raw_vx = np.random.randn(num_timesteps + 100)
    raw_vy = np.random.randn(num_timesteps + 100)

    smooth_vx = np.convolve(raw_vx, np.ones(100) / 100, mode='same')[50:-50]
    smooth_vy = np.convolve(raw_vy, np.ones(100) / 100, mode='same')[50:-50]

    velocities = np.vstack((smooth_vx, smooth_vy)).T * 10.0  # Scale up for variance
    speeds = np.linalg.norm(velocities, axis=1)
    angles = np.arctan2(velocities[:, 1], velocities[:, 0])

    # 3. SPIKE GENERATION: Simulate Action Potentials (Poisson Process)
    print("Simulating Motor Cortex Spikes (Georgopoulos Cosine Tuning)...")
    spike_counts = np.zeros((num_timesteps, num_neurons), dtype=np.float32)

    for i in range(num_neurons):
        # Calculate expected firing rate (lambda) for this neuron at each timestep
        rate = baseline_rates[i] + modulation_depths[i] * speeds * np.cos(angles - preferred_directions[i])
        rate = np.clip(rate, 0.1, None)  # Firing rates cannot be negative

        # Expected spikes in this 50ms bin
        expected_spikes = rate * dt

        # Sample actual discrete spikes from a Poisson distribution
        spike_counts[:, i] = np.random.poisson(expected_spikes)

    # 4. Save the BCI Dataset
    os.makedirs("bci_data", exist_ok=True)
    file_path = "bci_data/synthetic_motor_cortex.npz"
    np.savez(file_path, spikes=spike_counts, velocities=velocities)

    print(f"\n✅ Success! Saved synthetic neural data to {file_path}")
    print(f"   Shape of Spike Data (X): {spike_counts.shape} -> (Timesteps, Channels)")
    print(f"   Shape of Kinematics (Y): {velocities.shape} -> (Timesteps, X/Y Velocity)")


if __name__ == "__main__":
    main()