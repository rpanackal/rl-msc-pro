from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RandomCropSequence(nn.Module):
    """Crop the given sequence at a random location. Padding can be forced with paddding
    parameter. If crop size is larger than sequence then, padding is done on the left of sequence
    if pad_if_needed is True.
    
    She sequence is is expected to have [seq_length, feat_dim] shape, 
    where ... means an arbitrary number of leading dimensions, but if non-constant padding is used, 
    the input is expected to have at most 2 leading dimensions

    #! Arbitrary leading dimensions not handles atm.

    Args:
        length (int): Desired output length of the crop.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None. If a single int is provided this
            can be used to pad all borders. If sequence of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a sequence of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.

            .. note::
                In torchscript mode padding as single int is not supported, use a sequence of
                length 1: ``[padding, ]``.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill (number or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
            Only number is supported for torch Tensor.
            Only int or tuple value is supported for PIL Image.
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value at the edge of the image.
              If input a 5D torch Tensor, the last 3 dimensions will be padded instead of the last 2

            - reflect: pads with reflection of image without repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
              will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
              will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """
    def __init__(
        self,
        length: int,
        padding=None,
        pad_if_needed=False,
        fill=0,
        padding_mode="constant",
    ):
        super().__init__()
        self.length = length

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, seq):
        """
        Args:
            seq (Tensor): Sequence to be cropped.
                shape: (seq_length, feat_dim)

        Returns:
            Tensor: Cropped sequence.
                shape: (seq_length, feat_dim)
        """
        seq = seq.permute(1, 0).contiguous() # (feat_dim, seq_length)

        if self.padding is not None:
            seq = F.pad(seq, self.padding, value=self.fill, mode=self.padding_mode)

        _, seq_length = seq.size()

        # pad the width if needed
        if self.pad_if_needed and seq_length < self.length:
            padding = [self.length - seq_length, 0]
            seq = F.pad(seq, padding, value=self.fill, mode=self.padding_mode)
            return seq.permute(1, 0).contiguous()

        i = torch.randint(0, seq_length - self.length + 1, (1, ))
        seq = seq[:, i: i + self.length]

        return seq.permute(1, 0).contiguous()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.length}, padding={self.padding})"


if __name__ == "__main__":
    a = torch.arange(20).reshape(10, 2)
    print(f"a: {a}")

    crop_length_1 = 5
    crop = RandomCropSequence(crop_length_1)
    print(f"Cropped from a: {crop(a)}")

    crop_length_2 = 15
    crop_pad = RandomCropSequence(crop_length_2, pad_if_needed=True)
    print(f"Cropped from a: {crop_pad(a)}")