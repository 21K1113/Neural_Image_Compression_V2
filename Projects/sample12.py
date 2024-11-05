import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Define the decoder network
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, 1)

    def forward(self, z, x):
        z = z.expand(x.size(0), -1)
        x = torch.cat([z, x], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        sdf = torch.tanh(self.fc8(x))
        return sdf


# Generate sample points from a sphere
def generate_sphere_samples(num_samples):
    angles = np.random.rand(num_samples, 2) * np.array([np.pi, 2 * np.pi])
    points = np.zeros((num_samples, 3))
    points[:, 0] = np.sin(angles[:, 0]) * np.cos(angles[:, 1])
    points[:, 1] = np.sin(angles[:, 0]) * np.sin(angles[:, 1])
    points[:, 2] = np.cos(angles[:, 0])
    return points


# Compute the SDF values for a sphere
def sphere_sdf(points):
    return np.linalg.norm(points, axis=1) - 1


def clamp(x, delta):
    return torch.minimum(torch.tensor([delta]), torch.maximum(x, torch.tensor([-delta])))


def customLoss(outputs, targets, delta=0.1):
    return torch.mean(torch.abs(clamp(outputs, delta) - clamp(targets, delta)))


# Hyperparameters
latent_dim = 64
hidden_dim = 128
num_samples = 320000
batch_size = 3200
num_epochs = 1000
learning_rate = 0.001

# Generate training data
points = generate_sphere_samples(num_samples)
sdf_values = sphere_sdf(points)

# Convert to PyTorch tensors
points = torch.tensor(points, dtype=torch.float32)
sdf_values = torch.tensor(sdf_values, dtype=torch.float32).unsqueeze(-1)

# Initialize the latent code and the decoder
latent_code = torch.randn(1, latent_dim, requires_grad=True)
decoder = Decoder(latent_dim, hidden_dim)

# Optimizers
optimizer = optim.Adam([latent_code] + list(decoder.parameters()), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch in range(num_samples//batch_size):
        optimizer.zero_grad()
        batch_points = points[batch*batch_size:(batch+1)*batch_size]
        batch_sdf_values = sdf_values[batch*batch_size:(batch+1)*batch_size]
        batch_sdf_pred = decoder(latent_code, batch_points)
        loss = customLoss(batch_sdf_pred, batch_sdf_values)
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print("Training completed!")


# Visualization
def visualize_decoder(decoder, latent_code):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Generate a grid of points
    grid_points = np.linspace(-1.5, 1.5, 50)
    grid_x, grid_y, grid_z = np.meshgrid(grid_points, grid_points, grid_points)
    grid_coords = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)

    # Convert to PyTorch tensor
    grid_coords = torch.tensor(grid_coords, dtype=torch.float32)

    # Predict SDF values for the grid points
    with torch.no_grad():
        sdf_pred = decoder(latent_code, grid_coords).cpu().numpy()

    # Determine an appropriate threshold
    sdf_abs = np.abs(sdf_pred).flatten()
    sdf_abs_sort = np.sort(sdf_abs)
    threshold = sdf_abs_sort[5000]

    print(f"Determined threshold: {threshold}")

    # Extract surface points
    surface_points = grid_coords.numpy()[sdf_abs < threshold]

    # Plot the surface points
    ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], s=1)
    plt.show()


# Visualize the learned shape
visualize_decoder(decoder, latent_code)
