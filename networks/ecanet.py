### Import necessary dependencies
import torch
from torch import nn
from math import log

class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
    """
    
    def __init__(self, channel):
        super(ECA, self).__init__()
        k_size = int(abs((log(channel,2)+1)/2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        #print(y.shape)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = torch.sigmoid(y)
        #print("here",x.shape,y.shape)
        k=x * y.expand_as(x)
        #print("here",x.shape)
        #print(k.shape)
        return x * y.expand_as(x)