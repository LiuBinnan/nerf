import torch as th
import torch.nn as nn
import torch.nn.functional as F

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=True):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +  
            [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_ch, W) for i in range(D-1)]
        )
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.sigma_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
    
    def forward(self, x):
        pts, viewdirs = th.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = th.cat([pts, h], dim=-1)

        if self.use_viewdirs:
            sigma = self.sigma_linear(h)
            feature = self.feature_linear(h)
            h = th.cat([viewdirs, feature], dim=-1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = th.cat([sigma, rgb], dim=-1)
        else:
            outputs = self.output_linear(h)

        return outputs
