import torch
from torch import nn

class ResidualBlock(nn.Module):
  def __init__(self,n_features,ker_size=3):
    super().__init__()
    self.convStack = nn.Sequential(
        nn.Conv1d(in_channels=n_features,out_channels=n_features,kernel_size=ker_size,padding=1),
        nn.Tanh(),
        nn.Conv1d(in_channels=n_features,out_channels=n_features,kernel_size=ker_size,padding=1)
    )
  def forward(self,x):
    res = self.convStack(x)
    x = x + res
    return x
class Decoder(nn.Module):
  def __init__(self,n_features):
    super().__init__()
    self.s1f1 = nn.Conv1d(in_channels=1,out_channels=n_features,kernel_size=1,padding=0,stride=1)
    self.resStack = nn.Sequential(
        ResidualBlock(n_features),
        ResidualBlock(n_features),
        ResidualBlock(n_features),
        ResidualBlock(n_features),
        ResidualBlock(n_features),
        ResidualBlock(n_features),
        ResidualBlock(n_features),
        ResidualBlock(n_features),
        nn.Conv1d(in_channels=n_features,out_channels=n_features,kernel_size=3,padding=1)
    )
    self.cf4 = nn.Conv1d(in_channels=n_features,out_channels=n_features*4,kernel_size=3,padding=1,stride=1)
    self.upsample = nn.ConvTranspose1d(in_channels=n_features*4, out_channels=n_features*3, kernel_size=3, padding=1, stride=2, output_padding=1)
    self.A1 = nn.Conv1d(in_channels=n_features*3, out_channels=1, kernel_size=3, padding=1, stride=1)
    self.A2 = nn.Conv1d(in_channels=n_features*3, out_channels=1, kernel_size=3, padding=2, stride=1, dilation=2)
    self.A4 = nn.Conv1d(in_channels=n_features*3, out_channels=1, kernel_size=3, padding=4, stride=1, dilation=4)

  def forward(self,x,y):
    out_s1f1 = self.s1f1(x) + y
    res = self.resStack(out_s1f1)
    res = res + out_s1f1
    res = self.cf4(res)
    res = self.upsample(res)
    A1_out = self.A1(res)
    A2_out = self.A2(res)
    A4_out = self.A4(res)
    Fs = A1_out + A2_out + A4_out
    return Fs
  