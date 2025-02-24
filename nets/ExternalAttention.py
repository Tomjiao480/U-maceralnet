import numpy as np
import torch
from torch import nn
from torch.nn import init

class ExternalAttention(nn.Module):
    def __init__(self, in_channels, out_channels, S=64):
        super().__init__()
        self.mk = nn.Conv2d(in_channels, S, kernel_size=1, bias=False)
        self.mv = nn.Conv2d(S, out_channels, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        batch_size, num_channels, height, width = queries.size()
        attn = self.mk(queries)  # bs, S, h, w
        attn = attn.view(batch_size, attn.size(1), -1)  # bs, S, h*w
        attn = self.softmax(attn)  # bs, S, h*w
        attn = attn.view(batch_size, -1, height, width)  # bs, S, h, w
        out = self.mv(attn)  # bs, out_channels, h, w
        return out

# if __name__ == '__main__':
#     input = torch.randn(1, 512, 512, 512)  # Example input with batch_size=50, channels=512, height=7, width=7
#     ea = ExternalAttention(in_channels=512, out_channels=512, S=8)
#     output = ea(input)
#     print(output.shape)  # Should output: torch.Size([50, 512, 7, 7])
