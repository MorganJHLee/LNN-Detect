import torch
import torch.nn as nn
import random

class LiquidNode(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation_fn=nn.Tanh()):
        super(LiquidNode, self).__init__()
        self.hidden_dim = hidden_dim
        # Parameters controlling decay and influence of input
        self.decay = nn.Parameter(torch.normal(mean=0.05, std=0.01, size=(hidden_dim,)))  # Controls how much of the previous state is retained
        self.input_weight = nn.Parameter(torch.rand(hidden_dim) * 0.1)  # Controls how much influence the current input has
        self.activation_fn = activation_fn
        # Linear transformation to merge input and previous state
        self.linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        # Gating mechanism to regulate how new inputs affect the state
        self.gate = nn.Sigmoid()  # Sigmoid to generate gate values between 0 and 1

    def forward(self, x, prev_state):
        # Concatenate input and previous state to create combined input for linear transformation
        combined_input = torch.cat((x, prev_state), dim=1)
        # Apply linear transformation and add bias
        linear_output = self.linear(combined_input) + self.bias
        # Compute gate value to control the contribution of input and previous state
        gate_value = self.gate(linear_output)
        # Update the state using decay, input influence, and gate
        new_state = gate_value * (self.decay * prev_state + self.input_weight * linear_output)
        return self.activation_fn(new_state)

class LiquidLayer1(nn.Module):
    def __init__(self, input_dim, num_nodes, hidden_dim=10, activation_fn=nn.Tanh(), sparsity=0.2):
        super(LiquidLayer1, self).__init__()
        self.num_nodes = num_nodes
        # Create nodes for this layer
        self.nodes = nn.ModuleList([LiquidNode(input_dim, hidden_dim, activation_fn=activation_fn) for _ in range(num_nodes)])
        # Define sparse fixed random connections between nodes using an adjacency matrix
        self.connections = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and random.random() < sparsity:
                    self.connections[i, j] = True

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        # Pre-allocate tensor for outputs
        outputs = torch.zeros(batch_size, seq_len, self.num_nodes, self.nodes[0].hidden_dim, device=x.device)
        # Initialize previous states for each node
        prev_states = torch.zeros(self.num_nodes, batch_size, self.nodes[0].hidden_dim, device=x.device)

        # Iterate over each time step
        for t in range(seq_len):
            current_input = x[:, t, :]  # Get the input for the current time step
            current_states = []
            # Update each node's state based on the current input and its previous state
            for i, node in enumerate(self.nodes):
                prev_state = prev_states[i]
                # Aggregate contributions from connected nodes (echo state mechanism)
                connected_states = [prev_states[j] for j in range(self.num_nodes) if self.connections[i, j]]
                if connected_states:
                    aggregated_state = torch.stack(connected_states, dim=0).sum(dim=0)
                    prev_state += aggregated_state  # Add the aggregated states from connected nodes
                current_state = node(current_input, prev_state)
                current_states.append(current_state)
            # Update previous states for the next time step
            prev_states = torch.stack(current_states, dim=0)
            # Store output for the current time step
            outputs[:, t, :, :] = torch.stack(current_states, dim=1)

        return outputs

class LiquidLayer2(nn.Module):
    def __init__(self, input_dim, num_nodes, hidden_dim=10, activation_fn=nn.Tanh(), sparsity=0.2):
        super(LiquidLayer2, self).__init__()
        self.num_nodes = num_nodes
        # Create nodes for this layer
        self.nodes = nn.ModuleList([LiquidNode(input_dim, hidden_dim, activation_fn=activation_fn) for _ in range(num_nodes)])
        # Define sparse fixed random connections between nodes using an adjacency matrix
        self.connections = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and random.random() < sparsity:
                    self.connections[i, j] = True

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        # Pre-allocate tensor for outputs
        outputs = torch.zeros(batch_size, seq_len, self.num_nodes, self.nodes[0].hidden_dim, device=x.device)
        # Initialize previous states for each node
        prev_states = torch.zeros(self.num_nodes, batch_size, self.nodes[0].hidden_dim, device=x.device)

        # Iterate over each time step
        for t in range(seq_len):
            current_input = x[:, t, :]  # Get the input for the current time step
            current_states = []
            # Update each node's state based on the current input and its previous state
            for i, node in enumerate(self.nodes):
                prev_state = prev_states[i]
                # Aggregate contributions from connected nodes (echo state mechanism)
                connected_states = [prev_states[j] for j in range(self.num_nodes) if self.connections[i, j]]
                if connected_states:
                    aggregated_state = torch.stack(connected_states, dim=0).sum(dim=0)
                    prev_state += aggregated_state  # Add the aggregated states from connected nodes
                current_state = node(current_input, prev_state)
                current_states.append(current_state)
            # Update previous states for the next time step
            prev_states = torch.stack(current_states, dim=0)
            # Store output for the current time step
            outputs[:, t, :, :] = torch.stack(current_states, dim=1)

        return outputs

class AnomalyDetectionModule(nn.Module):
    def __init__(self, threshold=0.1):
        super(AnomalyDetectionModule, self).__init__()
        self.threshold = threshold

    def forward(self, input_data, baseline_data):
        # Calculate absolute deviation between input and baseline
        deviation = torch.abs(input_data - baseline_data)
        # Compute mean deviation across nodes and hidden dimensions
        deviation_score = torch.mean(deviation, dim=(2, 3))
        # Determine if deviation exceeds the threshold, indicating anomalies
        anomalies = deviation_score > self.threshold
        return deviation_score, anomalies

class LiquidAnomalyDetector(nn.Module):
    def __init__(self, input_dim, num_nodes_layer1, num_nodes_layer2, hidden_dim=10, activation_fn=nn.Tanh(), threshold=0.1):
        super(LiquidAnomalyDetector, self).__init__()
        # Define the first liquid layer with sparse random connections
        self.liquid_layer1 = LiquidLayer1(input_dim, num_nodes_layer1, hidden_dim, activation_fn)
        # Define the second liquid layer with sparse random connections
        self.liquid_layer2 = LiquidLayer2(num_nodes_layer1 * hidden_dim, num_nodes_layer2, hidden_dim, activation_fn)
        # Define the anomaly detection module
        self.anomaly_detection = AnomalyDetectionModule(threshold)

    def forward(self, x):
        # Pass input through the first liquid layer
        layer1_output = self.liquid_layer1(x)
        # Pool across nodes dimension to reduce complexity (mean pooling)
        layer1_output_pooled = torch.mean(layer1_output, dim=2)
        # Pass pooled output through the second liquid layer
        layer2_output = self.liquid_layer2(layer1_output_pooled)
        # Calculate deviation and identify anomalies
        deviation_score, anomalies = self.anomaly_detection(x, layer2_output)
        return deviation_score, anomalies

# Test Full Model
input_dim = 10
num_nodes_layer1 = 15
num_nodes_layer2 = 100
hidden_dim = 10
batch_size = 32
seq_len = 100

# Generate random input data for testing
test_input = torch.randn(batch_size, seq_len, input_dim)
# Instantiate the model
model = LiquidAnomalyDetector(input_dim, num_nodes_layer1, num_nodes_layer2, hidden_dim)
# Forward pass through the model
deviation_score, anomalies = model(test_input)
# Print output shapes to verify correctness
print(deviation_score.shape)  # Expected shape: (batch_size, seq_len)
print(anomalies.shape)  # Expected shape: (batch_size, seq_len)
