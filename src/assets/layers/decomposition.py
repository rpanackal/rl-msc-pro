import torch
import torch.nn as nn


class SeriesDecomposition(nn.Module):
    """A method of breaking down a time series into two systematic components:
    trend-cycle and seasonal variation.

    The trend component represents the long-term direction of the time series,
    which can be increasing, decreasing, or stable over time. The seasonal component
    represents the recurring patterns that occur within the time series, such as
    yearly or quarterly cycles.

    A moving average calculation is used to smoothing out short-term fluctuations
    and capturing the overall trend or underlying pattern in the data. The choice
    of kernel size can impact the level of smoothing and the granularity of the trend
    extracted from the original time series.

    *Note: Padding applied on both side prior to smoothing to preserve sequence length.
    """
    def __init__(self, kernel_size: int, *args, **kwargs) -> None:
        """Initialize a decomposition layer to aggregate the trend-cyclical part and
        extract the seasonal part from the series.

        Args:
            kernel_size (int): size of the window used for the average pooling to
                compute the trend component.
        """
        super().__init__(*args, **kwargs)

        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1, padding=0
        )  # moving average

    def forward(self, x: torch.FloatTensor):
        """_summary_

        Args:
            x (_type_): _description_
                shape: (batch_size, seq_length, embed_dim)

        Returns:
            tuple[2]: The seasonal and trend components of the series respectively.
                shape: (batch_size, seq_length, embed_dim) for both components.
        """
        x_padded = self.pad_series(x)

        # calculate the trend and seasonal part of the series
        x_trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        x_seasonal = x - x_trend

        return x_seasonal, x_trend

    def pad_series(self, x):
        """Padding on the both ends of the series along sequence dimension 
        with values at the sequence boundaries to preserve length after average 
        pooling.

        Args:
            x (_type_): _description_
                shape: (batch_size, seq_length, embed_dim)
        
        Returns:
            torch.FloatTensor: padded sequence
        """
        num_of_pads = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, num_of_pads, 1)
        end = x[:, -1:, :].repeat(1, num_of_pads, 1)

        return torch.cat([front, x, end], dim=1)
