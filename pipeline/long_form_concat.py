"""
Perform concatenation of short waveforms to long one, optionally
include overlap between different waveforms.

The source should have below architectures:
- wavs/
- data.csv
- spk2utt

We need spk2utt to determine whether concatenation should
be limited to each speaker or include short waveforms from
multiple speakers.
"""

import argparse
import os
import random
from collections import defaultdict

import pandas as pd
from pydub import AudioSegment


# Randomly generate combinations of bonafide and spoof wavs for concatenation
def create_random_combination(
    bonafide_wav_files,
    spoof_wav_files,
    wav2dur_file,
    out_data_dir,
    num_bonafides=2580,
    num_spoofs=22800,
    num_bonafides_single=3,
    num_spoofs_single=7,
):
    """
    Create random combination of the wav files and write metadata
    """
    comb_metadata = open(
        out_data_dir
        + "/src_comb_metadata_mc_{}_{}.txt".format(
            num_bonafides_single, num_spoofs_single
        ),
        "w",
    )

    # Load utterance durations from file
    wav2dur = {}
    with open(wav2dur_file, "r") as u:
        for line in u:
            wav, dur = line.split()
            wav2dur[wav] = dur

    # Generate bonafide combinations for long-form utterances
    num_bonafides_single_for_bonafide = num_bonafides_single + num_spoofs_single
    bonafide_wav_idx = 0
    while bonafide_wav_idx < num_bonafides:
        comb_bonafide_wav_name = "LA_bonafide_{}_{}".format(
            num_bonafides_single_for_bonafide, bonafide_wav_idx
        )
        comb_bonafide_wavs = random.sample(
            bonafide_wav_files, num_bonafides_single_for_bonafide
        )
        random.shuffle(comb_bonafide_wavs)
        comb_bonafide_utts = []
        comb_bonafide_durs = []
        comb_bonafide_labels = []
        for wav in comb_bonafide_wavs:
            # Append duration and label
            dur = wav2dur[wav]
            comb_bonafide_utts.append(wav)
            comb_bonafide_durs.append(str(dur))
            comb_bonafide_labels.append("b")
        comb_metadata.write(
            "{} {} {} {} bonafide\n".format(
                comb_bonafide_wav_name,
                ",".join(comb_bonafide_utts),
                ",".join(comb_bonafide_durs),
                ",".join(comb_bonafide_labels),
            )
        )
        bonafide_wav_idx += 1

    # Generate spoof combinations for long-form utterances
    spoof_wav_idx = 0
    while spoof_wav_idx < num_spoofs:
        comb_spoof_wav_name = "LA_spoof_{}_{}_{}".format(
            num_bonafides_single, num_spoofs_single, spoof_wav_idx
        )
        comb_spoof_wavs = random.sample(
            bonafide_wav_files, num_bonafides_single
        ) + random.sample(spoof_wav_files, num_spoofs_single)
        random.shuffle(comb_spoof_wavs)
        comb_spoof_utts = []
        comb_spoof_durs = []
        comb_spoof_labels = []
        for wav in comb_spoof_wavs:
            # Append duration and label
            dur = wav2dur[wav]
            comb_spoof_utts.append(wav)
            comb_spoof_durs.append(dur)
            label = "s" if wav in spoof_wav_files else "b"
            comb_spoof_labels.append(label)

        comb_metadata.write(
            "{} {} {} {} spoof\n".format(
                comb_spoof_wav_name,
                ",".join(comb_spoof_utts),
                ",".join(comb_spoof_durs),
                ",".join(comb_spoof_labels),
            )
        )
        spoof_wav_idx += 1
    print(
        "Metadata of list to create long-form files have been written to {}".format(
            out_data_dir
            + "/src_comb_metadata_mc_{}_{}.txt".format(
                num_bonafides_single, num_spoofs_single
            )
        )
    )
    comb_metadata.close()


# Generate combinations with speaker constraint (single-speaker mode)
def create_random_combination_single_spk(
    bonafide_wav_files,
    spoof_wav_files,
    spk2utt_file,
    utt2dur_file,
    out_data_dir,
    num_bonafides=2580,
    num_spoofs=22800,
    num_bonafides_single=3,
    num_spoofs_single=7,
):
    """
    Create random combination of wav files, with optional speaker constraint.
    """
    comb_metadata = open(
        os.path.join(
            out_data_dir,
            "src_comb_metadata_sc_{}_{}.txt".format(
                num_bonafides_single, num_spoofs_single
            ),
        ),
        "w",
    )

    # Load utterance durations
    utt2dur = {}
    with open(utt2dur_file, "r") as u:
        for line in u:
            utt, dur = line.split()
            utt2dur[utt] = dur

    # Load speaker-to-utterance mapping
    spk2utt = defaultdict(list)
    if spk2utt_file:
        with open(spk2utt_file, "r") as s:
            for line in s:
                parts = line.strip().split()
                spk = parts[0]
                utts = parts[1:]
                spk2utt[spk].extend(utts)

    # Index bonafide and spoof wavs by speaker
    if spk2utt_file:
        bonafide_by_spk = defaultdict(list)
        spoof_by_spk = defaultdict(list)

        for wav in bonafide_wav_files:
            utt = os.path.basename(wav).split(".")[0]
            for spk, utts in spk2utt.items():
                if utt in utts:
                    bonafide_by_spk[spk].append(wav)
                    break

        for wav in spoof_wav_files:
            utt = os.path.basename(wav).split(".")[0]
            for spk, utts in spk2utt.items():
                if utt in utts:
                    spoof_by_spk[spk].append(wav)
                    break
    else:
        bonafide_by_spk = {None: bonafide_wav_files}
        spoof_by_spk = {None: spoof_wav_files}

    # Create bonafide concatenation combinations by speaker
    bonafide_wav_idx = 0
    while bonafide_wav_idx < num_bonafides:
        spk = random.choice(list(bonafide_by_spk.keys()))
        if len(bonafide_by_spk[spk]) < num_bonafides_single + num_spoofs_single:
            continue  # Skip speakers with insufficient samples

        comb_bonafide_wav_name = "LA_bonafide_{}_{}".format(
            num_bonafides_single + num_spoofs_single, bonafide_wav_idx
        )
        comb_bonafide_wavs = random.sample(
            bonafide_wav_files, num_bonafides_single + num_spoofs_single
        )
        random.shuffle(comb_bonafide_wavs)
        comb_bonafide_utts = []
        comb_bonafide_durs = []
        comb_bonafide_labels = []
        for wav in comb_bonafide_wavs:
            dur = utt2dur[wav]
            comb_bonafide_utts.append(wav)
            comb_bonafide_durs.append(str(dur))
            comb_bonafide_labels.append("b")
        comb_metadata.write(
            "{} {} {} {} bonafide\n".format(
                comb_bonafide_wav_name,
                ",".join(comb_bonafide_utts),
                ",".join(comb_bonafide_durs),
                ",".join(comb_bonafide_labels),
            )
        )

        bonafide_wav_idx += 1

    # Create spoof concatenation combinations by speaker
    spoof_wav_idx = 0
    while spoof_wav_idx < num_spoofs:
        spk = random.choice(list(spk2utt.keys()))
        if (
            len(bonafide_by_spk[spk]) < num_bonafides_single
            or len(spoof_by_spk[spk]) < num_spoofs_single
        ):
            continue  # Skip speakers with insufficient samples

        comb_spoof_wav_name = "LA_spoof_{}_{}_{}".format(
            num_bonafides_single, num_spoofs_single, spoof_wav_idx
        )
        comb_spoof_wavs = random.sample(
            bonafide_wav_files, num_bonafides_single
        ) + random.sample(spoof_wav_files, num_spoofs_single)
        random.shuffle(comb_spoof_wavs)
        comb_spoof_utts = []
        comb_spoof_durs = []
        comb_spoof_labels = []
        for wav in comb_spoof_wavs:
            dur = utt2dur[wav]
            comb_spoof_utts.append(wav)
            comb_spoof_durs.append(dur)
            label = "s" if wav in spoof_wav_files else "b"
            comb_spoof_labels.append(label)

        comb_metadata.write(
            "{} {} {} {} spoof\n".format(
                comb_spoof_wav_name,
                ",".join(comb_spoof_utts),
                ",".join(comb_spoof_durs),
                ",".join(comb_spoof_labels),
            )
        )

        spoof_wav_idx += 1

    print(
        "Metadata written to {}".format(
            os.path.join(
                out_data_dir,
                "src_comb_metadata_sc_{}_{}.txt".format(
                    num_bonafides_single, num_spoofs_single
                ),
            )
        )
    )
    comb_metadata.close()


# Concatenate the list of wavs into a single long-form file
def concatenation_single(wav_paths, output_path):
    """
    Concatenate multiple wavs into one.
    """
    # Create empty AudioSegment and concatenate each wav
    concatenated_wav = AudioSegment.empty()

    for path in wav_paths:
        wav = AudioSegment.from_file(path)
        concatenated_wav += wav

    concatenated_wav.export(output_path, format="wav")


# Orchestrates concatenation according to generated metadata
def concatenation(
    src_data_dir,
    out_data_dir,
    src_data_df,
    num_bonafides_single,
    num_spoofs_single,
    single_speaker=False,
):
    """
    Perform concatenation according to the metadata file fetched.
    """
    print("Begin concatenating wav files.......")
    if single_speaker:
        src_comb_metadata = out_data_dir + "/src_comb_metadata_sc_{}_{}.txt".format(
            num_bonafides_single, num_spoofs_single
        )
    else:
        src_comb_metadata = out_data_dir + "/src_comb_metadata_mc_{}_{}.txt".format(
            num_bonafides_single, num_spoofs_single
        )
    out_trial_txt = out_data_dir + "/asvspoof2019_trials.txt"
    out_data_csv = out_data_dir + "/data.csv"
    out_data_df = pd.DataFrame(columns=src_data_df.columns)

    num_bonafide_concat_wavs = 0
    num_spoof_concat_wavs = 0
    with open(src_comb_metadata, "r") as s, open(out_trial_txt, "w") as w:
        for line in s:
            # Parse metadata line
            utt, concat_wav_paths, _, _, decision = line.split()
            concat_wav_paths_list = concat_wav_paths.split(",")

            wav_path_list = []
            for wav_path in concat_wav_paths_list:
                if os.path.exists(wav_path):
                    wav_path_list.append(wav_path)
                else:
                    print("{} doesn't exist in wav paths".format(wav_path))
                    continue

            # Concatenate and save the new long-form wav
            concat_wav_path = out_data_dir + "/wavs/{}.wav".format(utt)
            concatenation_single(wav_path_list, concat_wav_path)

            if decision == "bonafide":
                num_bonafide_concat_wavs += 1
            else:
                num_spoof_concat_wavs += 1

            w.write("{} {} - - {}\n".format(utt, utt, decision))

            speaker = "single" if single_speaker else "multi"
            out_data_df.loc[len(out_data_df)] = [
                concat_wav_path,
                decision,
                speaker,
                "longform",
            ]

    out_data_df.to_csv(out_data_dir + "/data.csv")

    print(
        "Concatenated {} wav files. Bonafide: {}, Spoof: {}".format(
            num_bonafide_concat_wavs + num_spoof_concat_wavs,
            num_bonafide_concat_wavs,
            num_spoof_concat_wavs,
        )
    )


# Main entry: handles argument parsing, directory setup, and runs full process
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in_data_dir", type=str, help="Input data directory", required=True
    )
    parser.add_argument(
        "--out_data_dir", type=str, help="Output data directory", required=True
    )

    parser.add_argument(
        "--single_speaker",
        action="store_true",
        help="Whether we only concat wavs from the same speaker",
    )

    # Number of bonafide and spoof wavs generated
    parser.add_argument("--num_bonafides", type=int, default=2580)
    parser.add_argument("--num_spoofs", type=int, default=22800)

    # number of bonafide and spoof short wavs in each long-form wav file
    parser.add_argument("--num_bonafides_single", type=int, default=3)
    parser.add_argument("--num_spoofs_single", type=int, default=7)

    args = parser.parse_args()

    in_data_dir = args.in_data_dir
    out_data_dir = args.out_data_dir
    for i in ["data.csv", "spk2utt", "wavs", "utt2dur"]:
        assert os.path.exists(in_data_dir + "/" + i)

    os.makedirs(out_data_dir + "/wavs", exist_ok=True)

    # Load dataframe
    in_data_df = pd.read_csv(in_data_dir + "/data.csv")
    in_data_df = in_data_df.drop(columns=["Unnamed: 0"])

    # Separate bonafide and spoof wav file lists
    bonafide_wav_files = in_data_df[in_data_df["label"] == "bonafide"]["file"].tolist()
    spoof_wav_files = in_data_df[in_data_df["label"] == "spoof"]["file"].tolist()

    # Create random combinations and metadata for concatenation
    if args.single_speaker:
        create_random_combination_single_spk(
            bonafide_wav_files,
            spoof_wav_files,
            in_data_dir + "/spk2utt",
            in_data_dir + "/utt2dur",
            out_data_dir,
            num_bonafides=args.num_bonafides,
            num_spoofs=args.num_spoofs,
            num_bonafides_single=args.num_bonafides_single,
            num_spoofs_single=args.num_spoofs_single,
        )
    else:
        create_random_combination(
            bonafide_wav_files,
            spoof_wav_files,
            in_data_dir + "/utt2dur",
            out_data_dir,
            num_bonafides=args.num_bonafides,
            num_spoofs=args.num_spoofs,
            num_bonafides_single=args.num_bonafides_single,
            num_spoofs_single=args.num_spoofs_single,
        )
    # Perform concatenation according to generated metadata
    concatenation(
        in_data_dir,
        out_data_dir,
        in_data_df,
        args.num_bonafides_single,
        args.num_spoofs_single,
        single_speaker=args.single_speaker,
    )


if __name__ == "__main__":
    main()
