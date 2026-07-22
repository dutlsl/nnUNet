"""
ConvLSTM Cell and Multi-Layer ConvLSTM module for temporal feature fusion.

The ConvLSTMCell replaces standard fully-connected LSTM gates with 2D convolutions,
preserving spatial structure while modeling temporal dependencies.

Reference: Shi et al., "Convolutional LSTM Network: A Machine Learning Approach
for Precipitation Nowcasting", NeurIPS 2015.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, List


class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell that processes one time step.

    Args:
        input_dim:  Number of input feature channels.
        hidden_dim: Number of hidden state channels.
        kernel_size: Spatial kernel size for the convolution (default: 3).
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2  # Same-padding to preserve spatial resolution

        # Single Conv2d computes all 4 gates (i, f, g, o) at once for efficiency
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for a single time step.

        Args:
            x:     [B, input_dim, H, W] — current frame features
            state: (h_prev, c_prev) each [B, hidden_dim, H, W], or None for t=0

        Returns:
            h_next: [B, hidden_dim, H, W] — output hidden state
            (h_next, c_next): updated state tuple for next time step
        """
        B, _, H, W = x.size()

        if state is None:
            h_prev = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
            c_prev = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        else:
            h_prev, c_prev = state

        # Concatenate input and previous hidden state along channel dim
        combined = torch.cat([x, h_prev], dim=1)  # [B, input_dim + hidden_dim, H, W]

        # Compute all 4 gates in one Conv2d pass
        gates = self.conv(combined)  # [B, 4 * hidden_dim, H, W]

        # Split into individual gates
        cc_i, cc_f, cc_g, cc_o = torch.split(gates, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)   # Input gate:  how much new info to accept
        f = torch.sigmoid(cc_f)   # Forget gate: how much old memory to retain
        g = torch.tanh(cc_g)      # Candidate:   new information content
        o = torch.sigmoid(cc_o)   # Output gate: how much to expose

        # Update cell state (long-term memory) and hidden state (short-term output)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, (h_next, c_next)


class ConvLSTM(nn.Module):
    """
    Multi-step ConvLSTM that processes a sequence of T feature maps.

    Wraps ConvLSTMCell to iterate over the temporal dimension.
    Supports both training mode (full sequence) and streaming mode (single step).

    Args:
        input_dim:  Number of input feature channels.
        hidden_dim: Number of hidden state channels.
        kernel_size: Spatial kernel size (default: 3).
        num_layers: Number of stacked ConvLSTM layers (default: 1).
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, num_layers: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        layers = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            layers.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size))
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x_seq: torch.Tensor,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Process a temporal sequence of feature maps.

        Args:
            x_seq:  [B, T, C, H, W] — sequence of T feature maps
            states: List of (h, c) tuples per layer, or None

        Returns:
            output:     [B, hidden_dim, H, W] — hidden state of last time step (last layer)
            new_states: Updated list of (h, c) tuples per layer
        """
        B, T, C, H, W = x_seq.size()

        if states is None:
            states = [None] * self.num_layers

        # Iterate over time steps
        for t in range(T):
            x_t = x_seq[:, t]  # [B, C, H, W]

            new_states = []
            for layer_idx, cell in enumerate(self.layers):
                x_t, state = cell(x_t, states[layer_idx])
                new_states.append(state)

            states = new_states

        # Return final hidden state from last layer, and all layer states
        return x_t, states
