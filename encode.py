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



# def read_audio(file, device):
#    with wave.open(file, mode=None) as f:
#         audio_params = f.getparams()
#         print(audio_params)
#         raw_audio = f.readframes(audio_params[3])
#         # raw_audio = f.readframes(80000)
#         audio = np.frombuffer(raw_audio,dtype="int16")
#         print(audio.shape)
#         max_val=np.array(audio.max().astype("float32"))
#         print("Before normalizing: ",audio.min(),audio.max())
#         audio = normalize_segment(audio).astype("float32")
#         print("After normalizing: ",audio.min(),audio.max())
#         audio = torch.from_numpy(audio).unsqueeze(0).to(device)
#         return (audio,audio_params),max_val
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
    # print(type(file_path),file_path.split(".")[-1].upper(), file_path.split("/")[-1])
    waveform, sample_rate = torchaudio.load(file_path)
    num_channels = waveform.shape[0]
    max_val = torch.max(waveform)
    waveform = pad_audio(waveform,sample_rate)
    # print("Padded", waveform.shape)
    waveform = torch.flatten(waveform)
    # print("Flattened", waveform.shape)
    print("Normalizing Audio\n")
    waveform = normalize_torch(waveform).unsqueeze(0).to(device)
    # print("Normalized", waveform.shape)
    
    # Get properties of the audio file
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
    print(torch.__version__,device)
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
    # audio,audio_params = audio_data
    audio=audio_data
    enc_outputs = []
    x = audio
    for i in range(scale):
        x, res = model(x)
        if x.shape[-1]%2 == 1:
           x = torch.nn.functional.pad(x, (0, 1), mode='constant', value=0)
           res = torch.nn.functional.pad(res, (0, 1), mode='constant', value=0)
        print(x.shape,res.shape,x,res)
        enc_outputs.append(res)
    write_files(enc_outputs, audio_params, max_val)

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