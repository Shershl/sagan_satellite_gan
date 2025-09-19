import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, return_attention=False):
        B, C, H, W = x.size()

        # Query, Key, Value
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # B x (HW) x C
        proj_key = self.key_conv(x).view(B, -1, H * W)  # B x C x (HW)
        energy = torch.bmm(proj_query, proj_key)  # B x (HW) x (HW)
        attention = self.softmax(energy)  # увага між пікселями
        proj_value = self.value_conv(x).view(B, -1, H * W)  # B x C x (HW)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x (HW)
        out = out.view(B, C, H, W)
        out = self.gamma * out + x

        if return_attention:
            return out, attention.view(B, H * W, H, W)  # можна змінити shape при потребі
        return out
