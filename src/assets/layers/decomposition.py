import torch
import torch.nn as nn


class SeriesDecomposition(nn.Module):
    """
    A method of breaking down a time series into two systematic components:
    trend-cycle and seasonal variation.

    The trend component represents the long-term direction of the time series,
    which can be increasing, decreasing, or stable over time. The seasonal component
    represents the recurring patterns that occur within the time series, such as
    yearly or quarterly cycles.
    """

    def __init__(self, kernel_size, *args, **kwargs) -> None:
        """Initialize a decomposition layer to aggregate the trend-cyclical part and
        extract the seasonal part from the series

        Args:
            kernel_size (_type_): _description_
        """
        super().__init__(*args, **kwargs)

        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1, padding=0
        )  # moving average

    def forward(self, x: torch.FloatTensor):
        """_summary_

        Args:
            x (_type_): _description_
                shape: (batch_size, seq_length, embed_dim)

        Returns:
            _type_: _description_
        """
        x_padded = self.pad_series(x)

        # calculate the trend and seasonal part of the series
        x_trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        x_seasonal = x - x_trend

        return x_seasonal, x_trend

    def pad_series(self, x):
        """Padding on the both ends of the series to keep
        shape after average pooling.

        Args:
            x (_type_): _description_
        """
        num_of_pads = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, num_of_pads, 1)
        end = x[:, -1:, :].repeat(1, num_of_pads, 1)

        return torch.cat([front, x, end], dim=1)
