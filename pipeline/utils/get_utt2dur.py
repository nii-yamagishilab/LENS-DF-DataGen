"""
Get the utt2dur file
"""

import os
import sys
import pandas as pd
import librosa


# Main function: generate utt2dur file from wavs in input directory
def main():
    in_data_dir = sys.argv[1]
    for i in ["data.csv", "wavs"]:
        assert os.path.exists(in_data_dir + "/" + i)

    # Define paths for input CSV and output utt2dur
    csv_file = in_data_dir + "/data.csv"
    output_file = in_data_dir + "/utt2dur"

    # Load file list from CSV
    df = pd.read_csv(csv_file)

    with open(output_file, "w") as f:
        for file in df["file"]:
            # Check if the wav file exists
            if not os.path.exists(file):
                print("{} was not generated successfully in source".format(file))
                continue

            # Load wav file and calculate duration
            y, sr = librosa.load(file, sr=None)
            dur = len(y) / sr
            # Write to utt2dur file
            f.write(f"{file} {dur:.3f}\n")

    print("Finish creating utt2dur file for {}".format(in_data_dir))


if __name__ == "__main__":
    main()
