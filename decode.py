import torch
import torchaudio
import os
from torch import nn
import wave
import numpy as np
import argparse
from models.decoder import Decoder,ResidualBlock
from models.ansr import ASNR,SpatialAttention,SRBlock,SuperRes,MultiKernelConv
decoder_outputs = []

  
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models_path', type=str, help='Path to encoder file')
    args = parser.parse_args()
    delimiter="/"
    if delimiter in args.models_path:
       delimiter=""
    else:
       delimiter="/"
    decoder_path = args.models_path+delimiter+"decoder.pt"
    enc_output_path = "compressed_files/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"working on {device}")
    scale = 3
    max_val = torch.load(os.path.join(enc_output_path,"max_vals.pt"),map_location="cpu").to(device)
    enc_outputs = get_enc_outputs(enc_output_path, scale, device)
    decoder = torch.load(decoder_path,map_location="cpu").to(device)
    audio = decompress_audio(decoder, enc_outputs, scale, device)
    generate_audio(audio, enc_output_path,device)
    # generate_decoder_audio(enc_output_path, scale)
    print("files stored in decoder_outputs directory")

def get_enc_outputs(enc_output_path, scale, device):
    enc_outputs = []
    print("Retrieving Encoder Outputs")
    for i in range(scale,0,-1):
        x = torch.load(os.path.join(enc_output_path,f"enc_output_{i}.pt"),map_location="cpu").to(device)
        print(x.shape, x)
        enc_outputs.append(x)
    return enc_outputs


def min_max_normalize(x, x_min, x_max, a=-1, b=1):
    l=[]
    for element in x.detach().numpy():
        if element>0:
            l.append(((element - x_min) / (x_max - x_min)) * (b - a) + a)
        else:
            l.append((element-x_min)/(x_min-x_max)*(b-a)+a)
    return torch.tensor(l)

def decompress_audio(decoder, enc_outputs, scale, device):
    final = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=1,padding=0,stride=1).to(device)
    final.load_state_dict(torch.load("model_weights/final.pt"), strict=False)
    final_activation=nn.Tanh()
    prev_output = torch.zeros_like(enc_outputs[0]).to(device)
    print("Decompressing")
    for i in range(scale):
        print(enc_outputs[i].shape,prev_output.shape)
        if prev_output.shape[-1] != enc_outputs[i].shape[-1]:
           prev_output = prev_output[:,:-2]
        prev_output = decoder(enc_outputs[i],prev_output)
        decoder_outputs.append(final(prev_output))

    final_out = final(prev_output)
    return final_out

def normalize_segment(segment):
    max_val = np.max(np.abs(segment))
    return segment/max_val

# def generate_audio(audio, param_path):
#     audio = audio.squeeze()
#     audio_params = torch.load(os.path.join(param_path,"audio_params.pt"))
#     max_val = torch.load(os.path.join(param_path,"max_vals.pt")).item()
#     audio=(audio*max_val)
#     audio=audio.short()
#     with wave.open('decoder_outputs/output_audio.wav', 'wb') as f:
#         f.setparams(audio_params)
#         audio = audio.cpu()
#         audio = audio.detach()
#         audio = audio.numpy()
#         f.writeframes(audio.tobytes())

def generate_audio(audio, param_path,device):
    print("Generating Audio")
    sr=torch.load("model_weights/sr_v1.pt",map_location="cpu").to(device)
    sr_audio=sr(audio.unsqueeze(0))
    audio = audio.squeeze(0).cpu().detach()
    sr_audio = sr_audio.squeeze(0).cpu().detach()
    print(torch.load(os.path.join(param_path,"audio_params.pt")))
    sample_rate, num_channels, bit_depth, format, max_val, filename,format = torch.load(os.path.join(param_path,"audio_params.pt"))
    sr_output_path="decoder_outputs/reconstructed_sr_"+filename
    output_path ='decoder_outputs/reconstructed_'+filename
    # sr_output_path='decoder_outputs/sr_output'+filename
    max_val = max_val.cpu()
    if num_channels == 2:
       audio = audio.view(2,-1)
    #    sr_audio = sr_audio.view(2,-1)
    else:
       audio = torch.flatten(audio).unsqueeze(0)
    #    sr_audio = torch.flatten(sr_audio).unsqueeze(0)
    audio=(audio*max_val)
    info = torch.finfo(bit_depth)
    print(f"Audio params: {sample_rate}, {num_channels}, {bit_depth}, {format}, {info.bits }\n")
    torchaudio.save(output_path, audio, sample_rate=sample_rate, bits_per_sample=16,format=format)
    print(f"Audio saved to {output_path}\n")
    torchaudio.save(sr_output_path, sr_audio, sample_rate=sample_rate*4, bits_per_sample=16,format=format)
    print(f" SR Audio saved to {sr_output_path}\n")


if __name__ == "__main__":
    main()
