"""
Perform silence trimming and volume normalization on the flac/wav file
(Or any wav format that is readable by LibROSA)
"""

import argparse
import os
import shutil

import pandas as pd
import librosa
import soundfile as sf


def remove_silence_single(
    input_file,
    output_file,
    silence_threshold=-40,
    frame_length=2048,
    hop_length=512,
    sr=16000,
):
    """
    Removes silence from the beginning and end of an audio file efficiently.
    """

    # Load audio file
    audio, sample_rate = librosa.load(input_file, sr=sr)

    # Split the audio into non-silent intervals
    non_silent_intervals = librosa.effects.split(
        audio,
        top_db=-silence_threshold,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    # Efficiently find the first and last non-silent frames
    if len(non_silent_intervals) > 0:
        # Get the start of the first non-silent segment
        start = non_silent_intervals[0][0]
        # Get the end of the last non-silent segment
        end = non_silent_intervals[-1][1]
    else:
        # If no non-silent part is found, we assume it's all silence, return an empty audio
        start, end = 0, 0

    # Slice the non-silent part of the audio
    trimmed_audio = audio[start:end]

    # Save the processed audio to the output file
    sf.write(output_file, trimmed_audio, sample_rate)


def adjust_volume_sv56_single(input_wav_file, output_wav_file):
    """
    Adjust the volume of the waveform by sv56 toolkit
    """
    level_norm = 20
    os.system(
        "bash pipeline/utils/sub_sv56.sh {} {} {} 2>/dev/null".format(
            input_wav_file, output_wav_file, level_norm
        )
    )
    if not os.path.exists(output_wav_file):
        shutil.copyfile(input_wav_file, output_wav_file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in_data_dir", type=str, help="Input data directory", required=True
    )
    parser.add_argument(
        "--out_data_dir", type=str, help="Output data directory", required=True
    )

    args = parser.parse_args()

    in_data_dir = args.in_data_dir
    out_data_dir = args.out_data_dir
    for i in ["data.csv", "spk2utt"]:
        assert os.path.exists(in_data_dir + "/" + i)

    os.makedirs(out_data_dir + "/wavs", exist_ok=True)

    # perform the pre processing (silence trimming + volume normalization)
    # on the input raw audio
    # load the PD dataframe first
    in_data_df = pd.read_csv(in_data_dir + "/data.csv")

    # Create an empty DataFrame with the same columns
    out_data_df = pd.DataFrame(columns=in_data_df.columns)

    for index, row in in_data_df.iterrows():
        # define the file paths
        in_file_path = row["file"]
        wav_name = os.path.basename(in_file_path)
        out_file_path = out_data_dir + "/wavs/{}.wav".format(wav_name.split(".")[0])
        temp_file_path = out_data_dir + "/wavs/{}.wav".format(
            wav_name.split(".")[0] + "_temp"
        )

        # perform the normalization on single waveform
        remove_silence_single(in_file_path, temp_file_path)
        adjust_volume_sv56_single(temp_file_path, out_file_path)

        # copy the new file path and decision to the new CSV file
        new_row = row.copy()
        new_row["file"] = out_file_path
        out_data_df.loc[index] = new_row

    out_data_df.to_csv(out_data_dir + "/data.csv")
    shutil.copyfile(in_data_dir + "/spk2utt", out_data_dir + "/spk2utt")
    print(
        "Finish pre-processing wav files. New wav files are stored in {}".format(
            out_data_dir
        )
    )


if __name__ == "__main__":
    main()
