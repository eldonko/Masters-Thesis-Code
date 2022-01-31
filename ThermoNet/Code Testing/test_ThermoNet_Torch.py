from ThermoNet.Development.ThermoNetTorch import ThermoNet
import torch

net = ThermoNet()
inp = torch.randn(size=[10, 1])

print('in')
print(inp)

print('out')
print(net(inp))