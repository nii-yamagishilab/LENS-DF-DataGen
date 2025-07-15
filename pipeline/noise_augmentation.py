"""
Perform noise augmentation of particular direectory with below:
- wavs/
- data.csv
- spk2utt

The noise augmentation does NOT augment the data to N times more,
but randomly apply one type of noise onto the waveform. In such
case, the total amount of data is not augmented.
- This is to avoid the question that "does noise work or we simply
have more data to train?"

TODO We have disabled RIR for now.
"""

import argparse
import os
import sys
import random
import glob

import numpy as np
import pandas as pd
import shutil
import librosa
import soundfile as sf
from scipy import signal


MUSAN_DIR = "data/Database/musan"
RIR_DIR = "data/Database/RIRS_NOISES/simulated_rirs"


# Loader for MUSAN (noise) and RIR (reverberation) datasets
class rir_musan_loader(object):
    def __init__(self, musan_path, rir_path, samplerate=16000, snr_range=[0, 10]):
        # Initialize noise and RIR file lists and parameters
        self.noisetypes = ["noise", "speech", "music"]
        self.noisesnr = {"noise": snr_range, "speech": snr_range, "music": snr_range}
        self.numnoise = {"noise": [1, 1], "speech": [3, 8], "music": [1, 1]}

        self.noiselist = {}
        augment_files = glob.glob(os.path.join(musan_path, "*/*/*.wav"))
        for file in augment_files:
            noise_type = file.split("/")[-3]
            if noise_type not in self.noiselist:
                self.noiselist[noise_type] = []
            self.noiselist[noise_type].append(file)

        self.rir_files = glob.glob(os.path.join(rir_path, "*/*/*.wav"))
        self.samplerate = samplerate

    # Add a random type of noise or reverberation to input audio
    def add_noise(self, in_audio):
        noise_methods = ["-", "reverb", "babble", "music", "noise", "telnoise"]
        random_index = random.randint(0, 4)

        if random_index == 0:  # No augmentation
            audio = in_audio
        # elif random_index == 1:  # Reverberation (currently disabled)
        # audio = self.add_rev_single(in_audio)
        # audio = in_audio
        elif random_index == 1:  # Add babble noise (speech)
            audio = self.add_noise_single(in_audio, "speech")
        elif random_index == 2:  # Add music noise
            audio = self.add_noise_single(in_audio, "music")
        elif random_index == 3:  # Add generic noise
            audio = self.add_noise_single(in_audio, "noise")
        elif random_index == 4:  # Add both speech and music
            audio = self.add_noise_single(in_audio, "speech")
            audio = self.add_noise_single(audio, "music")

        return audio, noise_methods[random_index]

    # Apply reverberation using a random RIR file (not used in this script)
    def add_rev_single(self, audio):
        if len(self.rir_files) == 0:
            return audio  # Skip if no RIR files available
        rir_file = random.choice(self.rir_files)
        rir, sr = sf.read(rir_file)
        rir = np.expand_dims(rir.astype(np.float64), 0)
        rir = rir / np.sqrt(np.sum(rir**2) + 1e-8)  # Avoid divide-by-zero
        rir = np.squeeze(rir)
        return signal.convolve(audio, rir, mode="full")[: len(audio)]  # Maintain length

    # Add a single type of noise (speech, music, noise) to the audio
    def add_noise_single(self, audio, noisecat):
        if noisecat not in self.noiselist or len(self.noiselist[noisecat]) == 0:
            return audio  # Skip if no noise files

        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(
            self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1])
        )

        for noise in noiselist:
            noiseaudio, sr = sf.read(noise)
            if len(noiseaudio) == 0:
                continue  # Skip empty files

            # Ensure noise matches length of audio
            if len(noiseaudio) < len(audio):
                noiseaudio = np.tile(
                    noiseaudio, int(np.ceil(len(audio) / len(noiseaudio)))
                )
            noiseaudio = noiseaudio[: len(audio)]

            # Calculate RMS
            rms_audio = np.sqrt(np.mean(audio**2) + 1e-8)
            rms_noise = np.sqrt(np.mean(noiseaudio**2) + 1e-8)

            # SNR Control
            snr_range = self.noisesnr[noisecat]
            snr_db = random.uniform(snr_range[0], snr_range[1])

            snr_linear = 10 ** (snr_db / 20)
            desired_rms_noise = rms_audio / snr_linear

            # Scale noise to match target SNR
            scaled_noise = noiseaudio * (desired_rms_noise / rms_noise)

            # Add noise to audio
            audio = audio + scaled_noise
        return audio


# Simple unit test for the augmentation process
def unit_test():
    mock_loader = rir_musan_loader(MUSAN_DIR, RIR_DIR)

    audio, sr = librosa.load(sys.argv[1], sr=16000)

    # Apply augmentation
    augmented_audio, method = mock_loader.add_noise(audio)
    print(f"Augmentation Method: {method}")
    print(f"Input Length: {len(audio)}, Output Length: {len(augmented_audio)}")

    # Ensure no clipping occurs
    if np.max(np.abs(augmented_audio)) > 1.0:
        print("Warning: Clipping detected in augmented audio.")

    # Save augmented audio to 'test.wav'
    sf.write("test.wav", augmented_audio, sr)
    print("Augmented audio saved as 'test.wav'.")


# Wrapper for augmentation function (for external use)
def rir_musan_augmentation(augmenter, waveform):
    augmented_waveform, noise_type = augmenter.add_noise(waveform)
    return augmented_waveform, noise_type


# Main function: process a directory of wavs with augmentation
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in_data_dir", type=str, help="Input data directory", required=True
    )
    parser.add_argument(
        "--out_data_dir", type=str, help="Output data directory", required=True
    )
    parser.add_argument("--snr_range", type=str, required=True)

    args = parser.parse_args()

    in_data_dir = args.in_data_dir
    out_data_dir = args.out_data_dir
    for i in ["data.csv", "wavs", "utt2dur"]:
        assert os.path.exists(in_data_dir + "/" + i)

    os.makedirs(out_data_dir + "/wavs", exist_ok=True)

    # initialize the noise augmenter, with controllable SNR
    snr_range = [int(i) for i in args.snr_range.split("_")]
    noise_loader = rir_musan_loader(MUSAN_DIR, RIR_DIR, snr_range=snr_range)

    # Perform noise augmention on the waveform
    # load the input dataframe first
    in_data_df = pd.read_csv(in_data_dir + "/data.csv")
    in_data_df = in_data_df.drop(columns=["Unnamed: 0"])

    # Create an empty DataFrame with the same columns
    out_data_df = pd.DataFrame(columns=in_data_df.columns)

    for index, row in in_data_df.iterrows():
        # define the file paths
        in_file_path = row["file"]
        wav_name = os.path.basename(in_file_path)
        noise_file_path = out_data_dir + "/wavs/{}".format(wav_name)

        # TODO this is just in case there is a bug in the middle
        # of the generation, since the noise type is not that important
        # here, we skip it optionally
        noise_type = "-"
        # copy the new file path and decision to the new CSV file
        new_row = row.copy()
        if not os.path.exists(noise_file_path):
            input_audio, sr = librosa.load(in_file_path, sr=16000)
            input_audio, noise_type = noise_loader.add_noise(input_audio)
            sf.write(noise_file_path, input_audio, sr)
            new_row["attack"] = "longform-{}".format(noise_type)
        new_row["file"] = noise_file_path
        out_data_df.loc[index] = new_row

    out_data_df.to_csv(out_data_dir + "/data.csv")

    if os.path.exists(in_data_dir + "/spk2utt"):
        shutil.copyfile(in_data_dir + "/spk2utt", out_data_dir + "/spk2utt")

    print(
        "Finish performing noise augmentation. New wav files are stored in {}".format(
            out_data_dir
        )
    )


if __name__ == "__main__":
    # unit_test()
    main()
