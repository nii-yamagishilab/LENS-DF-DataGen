"""
Create subsample of ultra_deepfake.csv file for more effective evaluation
"""

import os
import argparse

import pandas as pd


# Main function: sample a subset of rows from CSV for evaluation
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in_data_dir", type=str, help="Input data directory", required=True
    )
    parser.add_argument("--num_subsamples", type=int, default=100000)
    args = parser.parse_args()

    in_data_dir = args.in_data_dir
    for i in ["ultra_deepfake.csv", "wavs"]:
        assert os.path.exists(in_data_dir + "/" + i)

    # Load the main CSV file
    df = pd.read_csv(in_data_dir + "/ultra_deepfake.csv")

    # Randomly sample N rows (default 100, can adjust)
    N = 100  # Change as needed
    sampled_df = df.sample(n=N, random_state=42)  # random_state for reproducibility

    # Save the sampled subset to a new CSV
    num_subsamples = args.num_subsamples
    sampled_csv = in_data_dir + "/ultra_deepfake_sample{}.csv".format(num_subsamples)
    sampled_df.to_csv(sampled_csv, index=False)

    print("Sampled {} rows and saved to {}".format(num_subsamples, sampled_csv))


if __name__ == "__main__":
    main()
