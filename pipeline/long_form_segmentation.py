"""
Perform segmentation on the waveforms with given segmentation
length, in order to perform experiments on duration
"""

import argparse
import os
import sys

import pandas as pd
import soundfile as sf
import librosa


def re_segmentation(
    src_id,
    src_wav_path,
    durations,
    labels,
    out_wav_dir,
    segment_length_seconds=4,
    decision="spoof",
    samplerate=16000,
):
    os.makedirs(out_wav_dir, exist_ok=True)

    concatenated_audio, _ = librosa.load(src_wav_path, sr=samplerate)

    # Segment the concatenated audio into chunks
    segment_samples = int(segment_length_seconds * samplerate)
    total_samples = len(concatenated_audio)

    start_idx = 0
    segment_idx = 1
    segment_metadata = []

    while start_idx < total_samples:
        end_idx = min(start_idx + segment_samples, total_samples)
        segment_audio = concatenated_audio[start_idx:end_idx]

        # Save the segmented audio
        segment_id = (
            f"{os.path.splitext(os.path.basename(src_wav_path))[0]}_{segment_idx}"
        )
        segment_path = os.path.join(out_wav_dir, f"{segment_id}.wav")
        sf.write(segment_path, segment_audio, samplerate)
        print(segment_path)

        # Store the segment for further processing
        segment_metadata.append((segment_id, start_idx, end_idx))

        # Update indices for the next chunk
        start_idx = end_idx
        segment_idx += 1

    # Step 2: Calculate spoof ratios and generate metadata
    segment_durations = [float(dur) for dur in durations.split(",")]
    segment_labels = labels.split(",")

    metadata = []
    segmented_trials_metadata = []
    current_chunk = []
    current_time = 0.0

    for duration, label in zip(segment_durations, segment_labels):
        while duration > 0:
            remaining_time = segment_length_seconds - current_time

            if duration >= remaining_time:
                # Add the remaining part to fill the current chunk
                current_chunk.append((remaining_time, label))
                duration -= remaining_time
                current_time = segment_length_seconds
            else:
                # Add the whole duration to the current chunk
                current_chunk.append((duration, label))
                current_time += duration
                duration = 0

            # If a chunk is complete
            if current_time >= segment_length_seconds:
                # Calculate the portion of 's' in the chunk
                portion_s = (
                    sum(d for d, l in current_chunk if l == "s")
                    / segment_length_seconds
                )

                decision = "spoof" if portion_s > 0.0 else "bonafide"

                metadata.append(f"{src_id}_{len(metadata) + 1} {portion_s} {decision}")
                segmented_trials_metadata.append(
                    f"{src_id}_{len(metadata) + 1} {src_id}_{len(metadata) + 1} - - {decision}"
                )

                # Reset for the next chunk
                current_chunk = []
                current_time = 0.0

    return metadata, segmented_trials_metadata


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in_data_dir", type=str, help="Input data directory", required=True
    )
    parser.add_argument(
        "--out_data_dir", type=str, help="Output data directory", required=True
    )
    parser.add_argument("--segment_length", type=float, default=4.0)

    args = parser.parse_args()

    in_data_dir = args.in_data_dir
    out_data_dir = args.out_data_dir
    for i in ["data.csv", "wavs"]:
        assert os.path.exists(in_data_dir + "/" + i)

    os.makedirs(out_data_dir + "/wavs", exist_ok=True)

    segment_length_seconds = args.segment_length

    # Perform noise augmention on the waveform
    # load the input dataframe first
    in_data_df = pd.read_csv(in_data_dir + "/data_sample.csv")
    in_data_df = in_data_df.drop(columns=["Unnamed: 0"])

    # Create an empty DataFrame with the same columns
    out_data_df = pd.DataFrame(columns=in_data_df.columns)

    # Segmentation with metadata and trial stats stored
    # read the original concatenation data for segmentation
    src_segment_file = "none"
    for filename in os.listdir(in_data_dir):
        if "src_comb_metadata" in filename:
            src_segment_file = in_data_dir + "/" + filename
    if src_segment_file == "none":
        sys.exit("Please check the original directory for the src_comb_metadata.txt")
    src_concat_wavs_dir = in_data_dir + "/wavs"
    out_segment_file = out_data_dir + "/segment_comb_metadata.txt"
    out_segment_wavs_dir = out_data_dir + "/wavs"
    out_segment_trials_file = out_data_dir + "/asvspoof2019_trials.txt"

    with open(src_segment_file, "r") as s, open(out_segment_file, "w") as t, open(
        out_segment_trials_file, "w"
    ) as tr:
        for line in s:
            concat_id, _, durations, labels, decision = line.split()
            src_concat_wav_path = src_concat_wavs_dir + "/{}.wav".format(concat_id)
            metadata, segmented_trials_metadata = re_segmentation(
                concat_id,
                src_concat_wav_path,
                durations,
                labels,
                out_segment_wavs_dir,
                segment_length_seconds=segment_length_seconds,
            )
            t.write("\n".join(metadata) + "\n")
            tr.write("\n".join(segmented_trials_metadata) + "\n")
            for item in metadata:
                utt_id, _, decision = item.split()
                wav_path = out_segment_wavs_dir + "/{}.wav".format(utt_id)
                out_data_df.loc[len(out_data_df)] = [
                    wav_path,
                    decision,
                    "segment_spk",
                    "longform",
                ]

    # Write the dataframe
    out_data_csv = out_data_dir + "/data.csv"
    out_data_df.to_csv(out_data_csv)


if __name__ == "__main__":
    main()
