import torch
import torch.nn as nn

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PredRNNCell(nn.Module):

    """

        This class implements a PredRNN cell for the Spatio-Temporal LSTM (ST-LSTM) architecture with dual memory states.
        The cell captures both spatial and temporal dependencies by maintaining two separate memory states:
        -   c:  Short-term memory for spatial information.
        -   m:  Long-term memory for temporal information.

        The implementation is based on the original paper:

        Wang, Y., Wu, H., Zhang, J., Gao, Z., Wang, J., Yu, P. S., & Long, M. (2021).
            PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning. ArXiv (Cornell University).
            https://doi.org/10.48550/arxiv.2103.09504
    
        Arguments:

            in_channels (int):  The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int):  The size of the convolutional kernel.
            padding (int):      The size of the padding.
            activation (str):   The activation function ("tanh" or "relu").

        Methods:

            forward (torch.Tensor): The forward pass of the PredRNN cell.

    """

    def __init__(
        self: None,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        padding: int,
        activation: str="tanh"
    ) -> None:
        super(PredRNNCell, self).__init__()
        self.input_channels=input_channels
        self.hidden_channels=hidden_channels
        self.kernel_size=kernel_size
        self.padding=padding
        if activation=="tanh":
            self.activation=torch.tanh
        elif activation=="relu":
            self.activation=torch.relu
        self.input_convolution=nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=5*self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=True
        )
        self.hidden_convolution=nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=5*self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=False
        )
        self.memory_convolution=nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=5*self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=False
        )

    """
    
        This method implements the forward pass of the PredRNN cell.
        
        Arguments:

            x (torch.Tensor):                       The input tensor.
            previous_hidden_state (torch.Tensor):   The hidden state of the previous cell.
            previous_cell_state (torch.Tensor):     The cell state of the previous cell.
            previous_memory_state (torch.Tensor):   The memory state of the previous cell.

        Returns:

            next_hidden_state (torch.Tensor):   The hidden state of the current cell.
            next_cell_state (torch.Tensor):     The cell state of the current cell.
            next_memory_state (torch.Tensor):   The memory state of the current cell.
    
    """

    def forward(
        self: None,
        x: torch.Tensor,
        previous_hidden_state: torch.Tensor,
        previous_cell_state: torch.Tensor,
        previous_memory_state: torch.Tensor
    ) -> tuple:
        combined=self.input_convolution(x)+self.hidden_convolution(previous_hidden_state)+self.memory_convolution(previous_memory_state)
        input_gate, forget_gate, output_gate, candidate_gate, reset_gate=torch.split(combined, self.hidden_channels, dim=1)
        input_gate=torch.sigmoid(input_gate)
        forget_gate=torch.sigmoid(forget_gate)
        output_gate=torch.sigmoid(output_gate)
        reset_gate=torch.sigmoid(reset_gate)
        candidate_gate=self.activation(candidate_gate)
        next_cell_state=forget_gate*previous_cell_state+input_gate*candidate_gate
        next_memory_state=reset_gate*previous_memory_state+(1-reset_gate)*next_cell_state
        next_hidden_state=output_gate*self.activation(next_cell_state+next_memory_state)
        return next_hidden_state, next_cell_state, next_memory_state

class PredRNN(nn.Module):

    """

        This class is a wrapper for the PredRNN cell.
        It stacks multiple PredRNN cells to create a deep PredRNN model for video frame prediction.

        Arguments:

            input_channels (int):   The number of input channels.
            hidden_channels (int):  The number of hidden channels in the PredRNN cells.
            kernel_size (int):      The size of the convolutional kernel.
            padding (int):          The size of the padding.
            n_layers (int):         The number of stacked PredRNN layers.
            activation (str):       The activation function ("tanh" or "relu").

        Methods:

            forward (torch.Tensor): The forward pass of the PredRNN model.

    """

    def __init__(
        self: None,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        padding: int,
        n_layers: int=2,
        activation: str="tanh"
    ) -> None:
        super(PredRNN, self).__init__()
        self.n_layers=n_layers
        self.hidden_channels=hidden_channels
        self.layers=nn.ModuleList([
            PredRNNCell(
                input_channels=input_channels if layer==0 else hidden_channels,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation
            )
            for layer in range(n_layers)
        ])
        self.output_convolution=nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=input_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    """

        This method implements the forward pass of the PredRNN model.

        Arguments:

            x (torch.Tensor):   The input tensor.

        Returns:

            output (torch.Tensor):  The predicted frame tensor.

    """

    def forward(
        self: None,
        x: torch.Tensor
    ) -> torch.Tensor:
        batch_size, sequence_length, _, height, width=x.size()
        device=x.device
        hidden_states=[torch.zeros(batch_size, self.hidden_channels, height, width, device=device) for _ in range(self.n_layers)]
        cell_states=[torch.zeros(batch_size, self.hidden_channels, height, width, device=device) for _ in range(self.n_layers)]
        memory_states=[torch.zeros(batch_size, self.hidden_channels, height, width, device=device) for _ in range(self.n_layers)]
        for t in range(sequence_length):
            current_input=x[:, t]
            for layer in range(self.n_layers):
                cell=self.layers[layer]
                hidden_layer, cell_layer, memory_layer=cell(
                    x=current_input,
                    previous_hidden_state=hidden_states[layer],
                    previous_cell_state=cell_states[layer],
                    previous_memory_state=memory_states[layer]
                )
                hidden_states[layer]=hidden_layer
                cell_states[layer]=cell_layer
                memory_states[layer]=memory_layer
                current_input=hidden_layer
        output=self.output_convolution(hidden_states[-1])
        output=torch.sigmoid(output)
        return output