import plotly.graph_objects as go  # Add Plotly import
from IPython.display import display
figure = None
def plot(x, y, predict=None):
    """
    Plot the original equation and the neural network's output.

    Args:
        x (torch.Tensor): Input values.
        y (torch.Tensor): Target values (original equation).
        predict (torch.Tensor): Neural network's output.
    """
    fig = go.FigureWidget()

    # Plot the original equation
    fig.add_trace(go.Scatter(
        x=x.squeeze().tolist(),
        y=y.squeeze().tolist(),
        mode='markers',  # Changed to dots
        name='Original Equation (y = 2x + 3)',
        marker=dict(color='blue'),
        showlegend=True  # Ensure legend is shown
    ))

    if predict is not None:
        # Plot the neural network's output
        fig.add_trace(go.Scatter(
            x=x.squeeze().tolist(),
            y=predict.detach().squeeze().tolist(),
            mode='markers',  # Changed to dots
            name='Neural Network Output',
            marker=dict(color='red'),
            showlegend=True  # Ensure legend is shown
        ))

    fig.update_layout(
        title="Regression of y = 2x + 3 Using Neural Network",
        xaxis_title="X",
        yaxis_title="y",
        legend_title="Legend",
        showlegend=True,  # Ensure legend is displayed

        xaxis_scaleanchor="y"  # Ensure X and Y axes are in the same scale
    )

    display(fig)

    global figure
    figure = Updatable(fig)

class Updatable():
    """
    A class to create an updatable object with a method to update its value.
    """
    def __init__(self, figObj: go.Figure):
        self.figObj = figObj

    def update(self, new_value):
        self.figObj.data[-1].y = new_value.detach().squeeze().tolist()