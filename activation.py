#%%
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go

# Define the function y = x^2 + 2
def target_function(x):
    return x**2 + 2

# Generate training data
x_train = torch.linspace(-5, 5, 100).unsqueeze(1)  # Shape: (100, 1)
y_train = target_function(x_train)

# Define a simple 2-layer neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Input layer to hidden layer
        self.relu = nn.ReLU()       # ReLU activation
        self.fc2 = nn.Linear(10, 1) # Hidden layer to output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Generate predictions
model.eval()
x_test = torch.linspace(-5, 5, 100).unsqueeze(1)
y_test = model(x_test).detach()

# Plot the original function and the model's predictions using Plotly
fig = go.Figure()

# Original function
fig.add_trace(go.Scatter(
    x=x_train.squeeze().tolist(),
    y=y_train.squeeze().tolist(),
    mode='lines',
    name='Original Function (y = x^2 + 2)',
    line=dict(color='blue')
))

# Predicted function
fig.add_trace(go.Scatter(
    x=x_test.squeeze().tolist(),
    y=y_test.squeeze().tolist(),
    mode='lines',
    name='Predicted Function',
    line=dict(color='red', dash='dash')
))

# Update layout
fig.update_layout(
    title="Function Regression with Neural Network",
    xaxis_title="x",
    yaxis_title="y",
    legend_title="Legend",
    showlegend=True
)

# Show the plot
fig.show()

# %%
