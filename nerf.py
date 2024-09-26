import torch as th
import numpy as np

from model import NeRF

def encode(x, L):
    fn = [lambda x, i: th.sin(th.tensor(2 ** i) * x), 
          lambda x, i: th.cos(th.tensor(2 ** i) * x)]
    res = [x]
    for i in range(L):
        for f in fn:
            res.append(f(x, i))
    return th.cat(res, dim=-1) #[N, 3*(2*L+1)]

if __name__ == '__main__':
    x = th.tensor([[0.3, 0.4, 0.5], [0.5, 0.4, 0.3]])
    x = encode(x, 3)
    net = NeRF(input_ch=x.shape[-1], input_ch_views=x.shape[-1])
    print(net)
    print(net(th.cat([x, x], dim=-1)))