from torch import nn
import torch

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
  
class Encoder(nn.Module):
  def __init__(self,n_features,out_channel=3):
    super().__init__()
    self.s2f5 = nn.Conv1d(in_channels=1,out_channels=n_features,kernel_size=5,padding=1,stride=2)
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
    self.s1f3 = nn.Conv1d(in_channels=n_features,out_channels=1,kernel_size=3,padding=1,stride=1)
    # self.s1f1 = nn.Conv1d(in_channels=n_features,out_channels=5,kernel_size=1,padding=1,stride=1)
    self.s1f1 = nn.Conv1d(in_channels=n_features,out_channels=1,kernel_size=3,padding=1,stride=1)


  def forward(self,x):
    x = nn.functional.pad(x, (0,1))
    out_s2f5 = self.s2f5(x)
    res = self.resStack(out_s2f5)
    res = res + out_s2f5
    Ein = self.s1f3(res)
    res = self.s1f1(res)
    return (Ein,res)
  
