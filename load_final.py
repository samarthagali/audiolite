from  torch import nn
import torch
from encoder import read_audio 
class ResidualBlock(nn.Module):
  def __init__(self,n_features,ker_size=3):
    super().__init__()
    self.convStack = nn.Sequential(
        nn.Conv1d(in_channels=n_features,out_channels=n_features,kernel_size=ker_size,padding=1),
        nn.ReLU(),
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
    print("encoder outputs:",Ein,res)
    return (Ein,res)

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

class L3C(nn.Module):
    def __init__(self, n_features, scale):
        super().__init__()
        self.scale = scale
        self.encoder = Encoder(n_features)
        self.decoder = Decoder(n_features)
        self.final = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=1,padding=0,stride=1)

    def forward(self, x):
        encoder_outputs = []
        residual_outputs = []
        for i in range(self.scale):
            x, residual = self.encoder(x)
            encoder_outputs.append(x)
            residual_outputs.append(residual)
        decoder_outputs = []
        prev_output = torch.zeros_like(residual_outputs[-1])
        for i in range(self.scale)[::-1]:
            prev_output = self.decoder(residual_outputs[i],prev_output)
            decoder_outputs.append(prev_output)
        final_out = self.final(decoder_outputs[-1])
        return final_out

device="cpu"
model=L3C(n_features=64,scale=3)
model.load_state_dict(torch.load("models/full_weights.pt",map_location="cpu"))
checkpoint=torch.load("models/full_weights.pt")
final_layer_state_dict = {k.split(".")[1]: v for k, v in checkpoint.items() if 'final' in k}
print(final_layer_state_dict)
torch.save(final_layer_state_dict,"results/final.pt")

audio_path = "audio_files/expected.wav"
audio_data,max_val = read_audio(audio_path,device)
scale = 3
audio,params=audio_data
output=model(audio)
print(output)