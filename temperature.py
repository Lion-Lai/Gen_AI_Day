#%%
import torch
import plotly.graph_objects as go

def apply_temperature(logits, temperature):
    """
    Adjust logits by applying temperature.
    """
    return logits / temperature

def softmax(logits):
    """
    Compute softmax probabilities from logits.
    """
    exp_logits = torch.exp(logits - torch.max(logits))
    return exp_logits / exp_logits.sum()

# Example logits (e.g., from a language model)
logits = torch.tensor([2.0, 1.0, 0.1])

# Temperature value (adjust this manually)
T = 1.0

# Compute probabilities before and after applying temperature
original_probs = softmax(logits)
adjusted_logits = apply_temperature(logits, T)
adjusted_probs = softmax(adjusted_logits)

# Plot the distributions
fig = go.Figure()
fig.add_trace(go.Bar(x=[f"Token {i}" for i in range(len(logits))], 
                     y=original_probs.numpy(), 
                     name="Original Distribution"))
fig.add_trace(go.Bar(x=[f"Token {i}" for i in range(len(logits))], 
                     y=adjusted_probs.numpy(), 
                     name=f"Adjusted Distribution (T={T})"))

fig.update_layout(title="Effect of Temperature on Sampling Distribution",
                  xaxis_title="Tokens",
                  yaxis_title="Probability",
                  barmode="group")
fig.show()

# %%
