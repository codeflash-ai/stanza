"""
A layer which implements maxout from the "Maxout Networks" paper

https://arxiv.org/pdf/1302.4389v4.pdf
Goodfellow, Warde-Farley, Mirza, Courville, Bengio

or a simpler explanation here:

https://stats.stackexchange.com/questions/129698/what-is-maxout-in-neural-network/298705#298705

The implementation here:
for k layers of maxout, in -> out channels, we make a single linear
  map of size in -> out*k
then we reshape the end to be (..., k, out)
and return the max over the k layers
"""


import torch
import torch.nn as nn

class MaxoutLinear(nn.Module):
    def __init__(self, in_channels, out_channels, maxout_k):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.maxout_k = maxout_k

        self.linear = nn.Linear(in_channels, out_channels * maxout_k)

    def forward(self, inputs):
        """
        Use the oversized linear as the repeated linear, then take the max

        One large linear map makes the implementation simpler and easier for pytorch to make parallel
        """
        # Combine view and max into a single reshape to avoid creating an extra view
        # Use torch.amax as a slightly more efficient alternative to torch.max along a dimension
        outputs = self.linear(inputs)
        # Instead of .view, use .reshape (same in this case) but move .reshape/.amax together for clarity (no copy for reshape if possible, contiguous anyway)
        shape = outputs.shape
        # Outputs shape: [..., out_channels * maxout_k] => [..., maxout_k, out_channels]
        outputs = outputs.reshape(*shape[:-1], self.maxout_k, self.out_channels)
        # torch.amax is marginally faster as it doesn't return indices
        outputs = torch.amax(outputs, dim=-2)
        return outputs

