import torch
import torch.nn as nn

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvLSTMCell(nn.Module):

    """
    
        This class implements a Convolutional LSTM (ConvLSTM) cell for spatio-temporal data processing.
        The ConvLSTM extends LSTM by incorporating convolutional operations.
        It uses a convolutional layer and gates (input, forget, output) to regulate information flow.
        It processes fixed-length sequences and captures spatial-temporal dependencies.

        The implementation is based on the original paper:

        Shi, X., Chen, Z., Wang, H., Yeung, D.-Y., Wong, W. K., & Woo, W. (2015).
            Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting.
            https://doi.org/10.48550/arxiv.1506.04214

        This implementation is adapted from the following sources:

        -   https://github.com/ndrplz/ConvLSTM_pytorch
        -   https://sladewinter.medium.com/video-frame-prediction-using-convlstm-network-in-pytorch-b5210a6ce582

        Arguments:

            in_channels (int):  The number of input channels.
            out_channels (int): The number of output channels.
            frame_size (tuple): The size of the input frame.
            kernel_size (int):  The size of the convolutional kernel.
            padding (int):      The size of the padding.
            activation (str):   The activation function ("tanh" or "relu").

        Methods:

            forward (torch.Tensor): The forward pass of the ConvLSTM cell.
    
    """

    def __init__(
        self: None,
        in_channels: int,
        out_channels: int,
        frame_size: tuple,
        kernel_size: int,
        padding: int,
        activation: str="tanh"
    ) -> None:
        super(ConvLSTMCell, self).__init__()  
        if activation=="tanh":
            self.activation=torch.tanh 
        elif activation=="relu":
            self.activation=torch.relu
        self.convolutional_layer=nn.Conv2d(
            in_channels=in_channels+out_channels, 
            out_channels=4*out_channels, 
            kernel_size=kernel_size, 
            padding=padding
        )
        self.input_gate_weights=nn.Parameter(data=torch.Tensor(out_channels, *frame_size))
        self.output_gate_weights=nn.Parameter(data=torch.Tensor(out_channels, *frame_size))
        self.forget_gate_weights=nn.Parameter(data=torch.Tensor(out_channels, *frame_size))

    """
    
        This method implements the forward pass of the ConvLSTM cell.
        
        Arguments:

            x (torch.Tensor):                       The input tensor.
            previous_hidden_state (torch.Tensor):   The hidden state of the previous cell.
            previous_cell_state (torch.Tensor):     The cell state of the previous cell.

        Returns:

            hidden_state (torch.Tensor):    The hidden state of the current cell.
            cell_state (torch.Tensor):      The cell state of the current cell.
    
    """

    def forward(
        self: None,
        x: torch.Tensor,
        previous_hidden_state: torch.Tensor,
        previous_cell_state: torch.Tensor
    ) -> tuple:
        convolutional_output=self.convolutional_layer(torch.cat(
                tensors=[x, previous_hidden_state],
                dim=1
            )
        )
        input_gate_convolution, forget_gate_convolution, cell_candidate_convolution, output_gate_convolution=torch.chunk(
            input=convolutional_output,
            chunks=4,
            dim=1
        )
        input_gate=torch.sigmoid(input=input_gate_convolution+self.input_gate_weights*previous_cell_state )
        forget_gate=torch.sigmoid(forget_gate_convolution+self.forget_gate_weights*previous_cell_state )
        cell_state=forget_gate*previous_cell_state+input_gate*self.activation(cell_candidate_convolution)
        output_gate=torch.sigmoid(output_gate_convolution+self.output_gate_weights*cell_state)
        hidden_state=output_gate*self.activation(cell_state)
        return hidden_state, cell_state
    
class ConvLSTM(nn.Module):

    """

        This class is a wrapper for the ConvLSTM cell.
        It processes sequences of fixed length and captures spatial-temporal dependencies.
        It uses a ConvLSTM cell to process the input tensor and generate the output tensor.

        Arguments:

            in_channels (int):  The number of input channels.
            out_channels (int): The number of output channels.
            frame_size (tuple): The size of the input frame.
            kernel_size (int):  The size of the convolutional kernel.
            padding (int):      The size of the padding.
            activation (str):   The activation function ("tanh" or "relu").

        Methods:

            forward (torch.Tensor): The forward pass of the ConvLSTM cell.

    """

    def __init__(
        self: None,
        in_channels: int,
        out_channels: int,
        frame_size: tuple,
        kernel_size: int,
        padding: int,
        activation: str="tanh"
    ) -> None:
        super(ConvLSTM, self).__init__()
        self.out_channels=out_channels
        self.convolutional_lstm_cell=ConvLSTMCell(
            in_channels=in_channels,
            out_channels=out_channels,
            frame_size=frame_size,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation
        )

    """

        This method is a wrapper for the forward pass of the ConvLSTM cell.

        Arguments:

            x (torch.Tensor):   The input tensor.

        Returns:

            output (torch.Tensor):  The output tensor.

    """

    def forward(
        self: None,
        x: torch.Tensor
    ) -> torch.Tensor:
        batch_size, _, sequence_length, height, width=x.size()
        output=torch.zeros(batch_size, self.out_channels, sequence_length, height, width, device=device)
        hidden_state=torch.zeros(batch_size, self.out_channels, height, width, device=device)
        cell_state=torch.zeros(batch_size,self.out_channels, height, width, device=device)
        for time_step in range(sequence_length):
            hidden_state, cell_state=self.convolutional_lstm_cell(x[:,:,time_step], hidden_state, cell_state)
            output[:,:,time_step]=hidden_state
        return output
    
class ConvLSTMNet(nn.Module):

    """

        This class implements a Seq2Seq model using stacked ConvLSTM layers.
        It captures spatio-temporal dependencies in sequential data, processes input sequences, and generates output sequences.
        It is suitable for video prediction, weather forecasting, and other spatio-temporal tasks.
        The architecture consists of multiple ConvLSTM layers followed by BatchNorm layers for regularisation.
        The final layer is a convolutional layer with a sigmoid activation function.

        Arguments:

            n_channels (int):   The number of input channels.
            n_kernels (int):    The number of kernels in the ConvLSTM layers.
            n_layers (int):     The number of ConvLSTM layers.
            kernel_size (int):  The size of the convolutional kernel.
            padding (int):      The size of the padding.
            activation (str):   The activation function ("tanh" or "relu").
            frame_size (tuple): The size of the input frame.

        Methods:

            forward (torch.Tensor): The forward pass of the ConvLSTMNet model.

    """

    def __init__(
        self: None,
        n_channels: int,
        n_kernels: int,
        n_layers: int,
        kernel_size: int,
        padding: int,
        activation: str,
        frame_size: tuple
    ) -> None:
        super(ConvLSTMNet, self).__init__()
        self.sequential=nn.Sequential()
        self.sequential.add_module(
            name="convlstm1",
            module=ConvLSTM(
                in_channels=n_channels,
                out_channels=n_kernels,
                frame_size=frame_size,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation
            )
        )
        self.sequential.add_module(
            name="batchnorm1",
            module=nn.BatchNorm3d(num_features=n_kernels)
        )
        for l in range(2, n_layers+1):
            self.sequential.add_module(
                name=f"convlstm{l}",
                module=ConvLSTM(
                    in_channels=n_kernels,
                    out_channels=n_kernels,
                    frame_size=frame_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    activation=activation
                )
            )
            self.sequential.add_module(
                name=f"batchnorm{l}",
                module=nn.BatchNorm3d(num_features=n_kernels)
            )
        self.convolutional_layer=nn.Conv2d(
            in_channels=n_kernels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    """

        This method implements the forward pass of the ConvLSTMNet model.

        Arguments:

            x (torch.Tensor):   The input tensor.

        Returns:

            output (torch.Tensor):  The output tensor.

    """

    def forward(
        self: None,
        x: torch.Tensor
    ) -> torch.Tensor:
        output=self.sequential(x)
        output=self.convolutional_layer(output[:,:,-1])
        return nn.Sigmoid()(output)