"""
From regular data.csv and the wav file, write the complete ultra deepfake CSV file.

The input directory shall simply have two files:
- data.csv
- utt2dur
- wavs/

Sample data.csv file:
,file,label,speaker,attack
0,data/asvspoof2019/LA/mc_p3/train/wavs/LA_bonafide_10_0.wav,bonafide,multi,longform
1,data/asvspoof2019/LA/mc_p3/train/wavs/LA_bonafide_10_1.wav,bonafide,multi,longform
2,data/asvspoof2019/LA/mc_p3/train/wavs/LA_bonafide_10_2.wav,bonafide,multi,longform

Sample target CSV file:
ID,Label,Duration,SampleRate,Path,Attack,Speaker,Proportion,AudioChannel,AudioEncoding,AudioBitSample,Language
Syn-0-S05M0359_0066597_0068124,real,1.53,16000,/home/smg/xuecliu/WORK/Project-Synthetiq-scripts/02-synthetiq-generator-jp/data/202408_Japanese_data/data/partition_2/bonafide/eval/wavs/S05M0359_0066597_0068124.wav,WORK,-,eval,1,PCM_S,16,JA
Syn-1-S08M1690_0604032_0606442,real,2.41,16000,/home/smg/xuecliu/WORK/Project-Synthetiq-scripts/02-synthetiq-generator-jp/data/202408_Japanese_data/data/partition_2/bonafide/eval/wavs/S08M1690_0604032_0606442.wav,WORK,-,eval,1,PCM_S,16,JA
"""

import argparse
import os
import sys

import pandas as pd
import soundfile as sf


def is_valid_audio(file_path: str) -> bool:
    """
    Checks if the given file is a valid WAV or FLAC file.

    :param file_path: Path to the audio file.
    :return: True if the file is a valid WAV or FLAC file, False otherwise.
    """
    if not os.path.isfile(file_path):
        return False

    if not (file_path.lower().endswith(".wav") or file_path.lower().endswith(".flac")):
        return False

    try:
        with sf.SoundFile(file_path) as audio_file:
            if audio_file.channels < 1 or audio_file.samplerate <= 0:
                return False
            return True
    except Exception:
        return False


def load_data(in_data_dir):
    data_csv_path = os.path.join(in_data_dir, "data.csv")
    utt2dur_path = os.path.join(in_data_dir, "utt2dur")

    if not os.path.exists(data_csv_path) or not os.path.exists(utt2dur_path):
        raise FileNotFoundError(
            "Required files data.csv or utt2dur not found in input directory."
        )

    data_df = pd.read_csv(data_csv_path)
    utt2dur_df = pd.read_csv(
        utt2dur_path,
        sep=" ",
        names=["file", "duration"],
        dtype={"file": str, "duration": float},
    )

    return data_df, utt2dur_df


def generate_output_csv(
    data_df, utt2dur_df, in_data_dir, sample_rate=16000, partition="train"
):
    output_rows = []
    wav_dir = os.path.join(in_data_dir, "wavs")

    # Preprocess durations into a dictionary for fast lookup
    file_to_duration = dict(zip(utt2dur_df["file"], utt2dur_df["duration"]))

    for _, row in data_df.iterrows():
        file_path = row["file"]
        # NOTE this is to compat with different version of protocols
        if row["label"] in ["bonafide", "real"]:
            label = "real"
        elif row["label"] in ["spoof", "fake"]:
            label = "fake"
        else:
            sys.exit("we have problems on labeling. Please check and do this again")
        speaker = row["speaker"]
        attack = row["attack"]

        # additional step: check the validity of the files
        abs_path = os.path.abspath(file_path)
        '''
        if not is_valid_audio(abs_path):
            print(
                f"Warning {file_path} is not a valid wav/flac file. Skip it.",
                file=sys.stderr,
            )
            continue
        '''

        # Get duration
        duration = file_to_duration.get(file_path)
        if duration is None:
            print(f"Warning: Duration not found for {file_path}", file=sys.stderr)
            continue

        file_name = os.path.basename(file_path).split(".")[0]
        output_rows.append(
            [
                f"Syn-{_}-{file_name}",
                label,
                duration,
                sample_rate,
                abs_path,
                attack,
                speaker,
                partition,
                1,
                "PCM_S",
                16,
                "EN",
            ]
        )

    output_df = pd.DataFrame(
        output_rows,
        columns=[
            "ID",
            "Label",
            "Duration",
            "SampleRate",
            "Path",
            "Attack",
            "Speaker",
            "Proportion",
            "AudioChannel",
            "AudioEncoding",
            "AudioBitSample",
            "Language",
        ],
    )

    output_path = os.path.join(in_data_dir, "ultra_deepfake.csv")
    output_df.to_csv(output_path, index=False)
    print("Output CSV saved to {}".format(output_path))


def main():
    parser = argparse.ArgumentParser(description="Generate Ultra Deepfake CSV file.")
    parser.add_argument(
        "--in_data_dir", type=str, help="Input data directory", required=True
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000, help="Sample rate (default: 16000)"
    )

    args = parser.parse_args()
    partition = "train" if "train" in args.in_data_dir else "eval"
    data_df, utt2dur_df = load_data(args.in_data_dir)
    generate_output_csv(
        data_df, utt2dur_df, args.in_data_dir, args.sample_rate, partition
    )


if __name__ == "__main__":
    main()
