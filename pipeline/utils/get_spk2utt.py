"""
Write spk2utt file based on data.csv
"""

import pandas as pd
import os
import sys


def write_spk2utt(data_csv_path):
    df = pd.read_csv(data_csv_path)

    # Group utterances by speaker
    spk2utts = {}
    for _, row in df.iterrows():
        spk = row["speaker"]
        wav_name = os.path.basename(row["file"])
        utt_id = wav_name.split(".")[0]
        if spk not in spk2utts:
            spk2utts[spk] = []
        spk2utts[spk].append(utt_id)

    # Write spk2utt file in the same directory as data.csv
    out_path = os.path.join(os.path.dirname(data_csv_path), "spk2utt")
    with open(out_path, "w") as f:
        for spk, utts in spk2utts.items():
            line = spk + " " + " ".join(utts) + "\n"
            f.write(line)
    print(f"spk2utt written to {out_path}")


if __name__ == "__main__":
    in_data_dir = sys.argv[1]
    for i in ["data.csv", "wavs"]:
        assert os.path.exists(in_data_dir + "/" + i)

    write_spk2utt(in_data_dir + "/data.csv")
