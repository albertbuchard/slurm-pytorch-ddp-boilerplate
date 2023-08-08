from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ConvParam:
    in_channels: int
    kernel_size: int
    padding: int
    stride: int
    pooling: Optional[int]
    out_channels: int


def compute_flattened_size(conv_params: List[ConvParam], input_width: int, input_height: int) -> int:
    """
    Calculate the flattened size after a series of convolution and pooling operations.

    Parameters:
    - conv_params: List of ConvParam objects.
    - input_width: Initial width of the image.
    - input_height: Initial height of the image.

    Returns:
    - The flattened size after all convolutions and poolings.
    """

    flattened_size = None
    for param in conv_params:
        input_width = (input_width + 2 * param.padding - param.kernel_size) // param.stride + 1
        input_height = (input_height + 2 * param.padding - param.kernel_size) // param.stride + 1

        # Assuming max pooling operation after the convolution
        if param.pooling is not None:
            input_width //= param.pooling
            input_height //= param.pooling

        # output
        flattened_size = input_width * input_height * param.out_channels

    return flattened_size

