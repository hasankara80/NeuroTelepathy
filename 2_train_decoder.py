import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os


# --- 1. DATA PREPARATION (The "History Window") ---
class BCIDataset(Dataset):
    def __init__(self, data_path, window_size=10, split='train'):
        # Load the synthetic brain data
        data = np.load(data_path)
        spikes = data['spikes']  # Shape: (Timesteps, 96 Channels)
        velocities = data['velocities']  # Shape: (Timesteps, 2 Dimensions)

        # Split into 80% Training, 20% Testing
        split_idx = int(len(spikes) * 0.8)
        if split == 'train':
            self.X = spikes[:split_idx]
            self.Y = velocities[:split_idx]
        else:
            self.X = spikes[split_idx:]
            self.Y = velocities[split_idx:]

        self.window_size = window_size

    def __len__(self):
        # We lose the first 'window_size' frames because we need history to predict the present
        return len(self.X) - self.window_size

    def __getitem__(self, idx):
        # Grab a window of past neural activity (e.g., 10 bins = 500ms of history)
        x_window = self.X[idx: idx + self.window_size]
        # Predict the velocity at the END of this window
        y_target = self.Y[idx + self.window_size - 1]

        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)


# --- 2. THE NEURAL DECODER MODEL ---
class LSTMDecoder(nn.Module):
    def __init__(self, num_channels=96, hidden_size=64, num_outputs=2):
        super(LSTMDecoder, self).__init__()
        # LSTM layer to process the sequence of spikes
        self.lstm = nn.LSTM(input_size=num_channels, hidden_size=hidden_size, batch_first=True)
        # Linear layer to map the LSTM memory to X and Y velocities
        self.fc = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        # x shape: (Batch, Window_Size, Channels)
        lstm_out, _ = self.lstm(x)

        # We only care about the LSTM's output at the final time step in the window
        last_timestep_out = lstm_out[:, -1, :]

        # Decode to velocity
        velocity_pred = self.fc(last_timestep_out)
        return velocity_pred


# --- 3. TRAINING AND EVALUATION ---
def main():
    print("Initializing Neuralink Decoder Training...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    data_path = "bci_data/synthetic_motor_cortex.npz"
    if not os.path.exists(data_path):
        raise FileNotFoundError("Data not found. Run 1_simulate_brain.py first.")

    # Load 500ms history windows (10 bins * 50ms)
    train_dataset = BCIDataset(data_path, window_size=10, split='train')
    test_dataset = BCIDataset(data_path, window_size=10, split='test')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = LSTMDecoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    epochs = 15
    print(f"Starting training on {len(train_dataset)} samples...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {total_loss / len(train_loader):.4f}")

    # --- 4. EVALUATION (The Metric Neuralink Cares About) ---
    print("\nEvaluating Decoder on unseen Test Data...")
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Pearson Correlation (Standard BCI Metric)
    corr_x = np.corrcoef(all_preds[:, 0], all_targets[:, 0])[0, 1]
    corr_y = np.corrcoef(all_preds[:, 1], all_targets[:, 1])[0, 1]

    print(f"✅ Decoding Complete!")
    print(f"📊 X-Velocity Correlation (R): {corr_x:.3f}")
    print(f"📊 Y-Velocity Correlation (R): {corr_y:.3f}")

    # Save the model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/bci_decoder.pt")

    # 5. Visual Proof
    print("\nGenerating visualization...")
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(all_targets[200:400, 0], label="True Intention (Human)", color='blue')
    plt.plot(all_preds[200:400, 0], label="AI Decoded (LSTM)", color='orange', linestyle='--')
    plt.title("X-Velocity Decoding over Time")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(all_targets[200:400, 1], label="True Intention (Human)", color='blue')
    plt.plot(all_preds[200:400, 1], label="AI Decoded (LSTM)", color='orange', linestyle='--')
    plt.title("Y-Velocity Decoding over Time")
    plt.legend()

    plt.tight_layout()
    plt.savefig("bci_data/decoding_results.png")
    print("Saved plot to bci_data/decoding_results.png")


if __name__ == "__main__":
    main()