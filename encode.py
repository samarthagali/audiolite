import torch
import torchaudio
import librosa
from torch import nn
import wave
import numpy as np
import os
import argparse
from models.encoder import Encoder,ResidualBlock
def check_and_create_dirs():
   if  not os.path.exists("compressed_files"):
      os.mkdir("compressed_files")
   if not os.path.exists("decoder_outputs"):
      os.mkdir("decoder_outputs")


def create_batches(audio, sample_length):
    # Reshape into batches of 5-second samples
    batched_audio = audio.view(-1, sample_length)  # Assuming dual channel audio
    return batched_audio
def pad_audio(tensor, frame_rate):
    # Get lengths of each channel
    channel_length = tensor.size(1)
    # frame_rate = 80000
    # Calculate padding for each channel
    padding = 0
    remainder = channel_length % frame_rate
    if remainder != 0:
        padding = frame_rate - remainder
    # print(frame_rate, padding, tensor.shape)
    # Pad each channel separately
    padded_tensor = torch.nn.functional.pad(tensor, (0, padding))

    return padded_tensor

def read_audio(file_path, device):
    # Read audio file using torchaudio
    print(f"Loading Audio: {file_path}\n")
    format = file_path.split(".")[-1]
    filename = file_path.split("/")[-1]
    waveform, sample_rate = torchaudio.load(file_path,format=format)
    num_channels = waveform.shape[0]
    max_val = torch.max(waveform)
    waveform = pad_audio(waveform,sample_rate*2)
    waveform = torch.flatten(waveform)
    print("Normalizing Audio\n")
    waveform = normalize_torch(waveform).unsqueeze(0).to(device)
    bit_depth = waveform.dtype
    return (waveform, (sample_rate, num_channels, bit_depth, format, max_val, filename))
def normalize_torch(segment):
    max_val = torch.max(torch.abs(segment))
    if max_val == 0:
        max_val = 1e-5
    if max_val == np.nan:
        max_val = 1e+7
    return segment/max_val

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on: ",device)
    parser = argparse.ArgumentParser()
    parser.add_argument('models_path', type=str, help='Path to models directory\n models should contain both encoder and decoder')
    parser.add_argument('audio_path', type=str, help='Path to audio file')
    args = parser.parse_args()
    args = parser.parse_args()
    delimiter="/"
    if delimiter in args.models_path:
       delimiter=""
    else:
       delimiter="/"
    encoder_path = args.models_path+delimiter+"encoder.pt"
    audio_path = args.audio_path
    audio_data = read_audio(audio_path,device)
    audio_data,params = audio_data[0],audio_data[1]
    max_val=params[4]
    print(f"audio params:{params}")
    encoder = torch.load(encoder_path,map_location=device).to(device)
    scale = 3
    print("compressing")
    args = gen_compressed_files(encoder,audio_data,params,scale,max_val)
    print("compressed files stored in compressed_files directory ")


def normalize_segment(segment):
    max_val = np.max(np.abs(segment))
    if max_val == 0:
        max_val = 1e-5
    if max_val == np.nan:
        max_val = 1e+7
    return segment/max_val





def gen_compressed_files(model, audio_data,audio_params, scale,max_val):
    check_and_create_dirs()
    audio= audio_data
    batch_length = audio_params[0]*2
    batched_audio = create_batches(audio, batch_length).unsqueeze(1)
    print(f"Original: {audio.shape}, Batched Shape:{batched_audio.shape}\n")
    enc_outputs = []
    x = batched_audio
    for i in range(scale):
        x2,res = [], []
        print(f"Scale {i+1}: {x.shape}\n")
        for input_batch in x:
            if input_batch.shape[-1]%2 == 1:
                input_batch = torch.nn.functional.pad(input_batch, (0, 1), mode='constant', value=0)
            out1, out2 = model(input_batch)
            x2.append(out1)
            res.append(out2)  
        x = torch.stack(x2)
        res = torch.stack(res)
           
        enc_outputs.append(res.to(torch.float16))
    write_files(enc_outputs, audio_params,max_val)
def write_files(files, audio_params, max_val):
    torch.save(max_val,"compressed_files/max_vals.pt")

    torch.save(audio_params,"compressed_files/audio_params.pt")

    # with open(".\\outputs\\audio_params.txt","w") as f:
    #    for i in audio_params:
    #       f.write(f"{i}\n")
   
    for i,file in enumerate(files):
       torch.save(file,f"compressed_files/enc_output_{i+1}.pt")

if __name__ == "__main__":
    main()