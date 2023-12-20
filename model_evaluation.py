import torch
import os
import torchaudio
import numpy as np
import pandas as pd
import subprocess
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def run_enc_dec(arg1,arg2):
    # # Check if two arguments are provided
    # if len(sys.argv) != 3:
    #     print(f"Usage: {sys.argv[0]} argument1 argument2")
    #     sys.exit(1)

    # # Assign command-line arguments to variables
    # arg1 = sys.argv[1]
    # arg2 = sys.argv[2]
    command = ["python", "encode.py", arg1, arg2]

    # Run the command
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(process.stdout,process.stderr )

    command = ["python", "decode.py", arg1]

    # Run the command
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(process.stdout,process.stderr )




def generate_audio_file(audio, filename, file_dir="intermediate_outputs/"):
    print("Generating Audio\n")
    audio = audio.squeeze().cpu().detach()
    sample_rate, num_channels, bit_depth, format, max_val, filename = torch.load(os.path.join("compressed_files/","audio_params.pt"))
    output_path = os.path.join(file_dir,filename)
    max_val = max_val.cpu()
    if num_channels == 2:
       audio = audio.view(2,-1)
    else:
       audio = torch.flatten(audio).unsqueeze(0)
    print(audio.shape, max_val)
    audio=(audio*max_val)
    print(audio)
    print(f"Audio params: {sample_rate}, {num_channels}, {bit_depth}, {format}, {bit_depth.itemsize * 8}\n")
    torchaudio.save(output_path, audio, sample_rate=sample_rate, bits_per_sample=bit_depth.itemsize * 8)
    print(f"Audio saved to {output_path}\n")

def perform_evaluation(samples_dir, models_path="model_weights/"):
    audio_files = [file for file in os.listdir(samples_dir) if file.endswith(".wav") or file.endswith(".mp3")]
    print(f"Audio files: {audio_files}")
    all_mse = []
    all_l1 = []
    all_psnr = []
    all_ssim = []

    for audio_file in audio_files:
        format = audio_file[-4:]
        audio_path = os.path.join(samples_dir,audio_file)
        output_path = 'decoder_outputs/reconstructed_' + audio_file 
        print("Paths:",audio_path,output_path)
        run_enc_dec(models_path, audio_path)
        input_audio, input_sr = torchaudio.load(audio_path)
        output_audio, output_sr = torchaudio.load(output_path)

        print("Shapes:",input_audio.shape, output_audio.shape)

        if input_audio.shape != output_audio.shape:
            input_audio = input_audio[:,:min(input_audio.shape[1],output_audio.shape[1])]
            output_audio = output_audio[:,:min(input_audio.shape[1],output_audio.shape[1])]

        print("Trimmed Shapes:",input_audio.shape, output_audio.shape)

        # Convert torch tensors to numpy arrays
        input_audio = input_audio.numpy()
        output_audio = output_audio.numpy()

        # Compute MSE and L1 loss
        mse_value = np.mean(np.square(input_audio - output_audio))
        l1_value = np.mean(np.abs(input_audio - output_audio))

        # Compute PSNR
        psnr_value = psnr(input_audio, output_audio, data_range=output_audio.max() - output_audio.min())

        # Compute SSIM
        # ssim_value, _ = ssim(input_audio, output_audio, full=True)

        # Append values to lists
        all_mse.append(mse_value.item())
        all_l1.append(l1_value.item())
        all_psnr.append(psnr_value)
        # all_ssim.append(ssim_value)

    print(audio_files,all_mse,all_l1, all_psnr)
    print(len(audio_files),len(all_mse),len(all_l1), len(all_psnr))
    # metrics_data = pd.DataFrame({"Audio File":audio_files, "MSE":all_mse,"L1 Loss":all_l1,"PSNR":all_psnr,"SSIM":all_ssim})
    metrics_data = pd.DataFrame({"Audio File":audio_files, "MSE":all_mse,"L1 Loss":all_l1,"PSNR":all_psnr})
    print(metrics_data)
    metrics_data.to_csv("SKIP_Connections_Metrics_Data.csv", index=False)

def main():
    samples_dir = "samples/"
    perform_evaluation(samples_dir)

if __name__ == "__main__":
    main()