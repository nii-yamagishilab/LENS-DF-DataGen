#!/bin/sh
#SBATCH --job-name=multi_channel
#SBATCH --out="log_multi_channel"
#SBATCH --time=72:00:00
#SBATCH -p qcpu

set -e
. ./path.sh

# Perform various level of processing and augmentation
# Single channel: Here we perform the concatenation first,
# then do the noise augmentation.
# - p1: Pre-processing
# - p2: Noise augmentation
# - p3: Long-form concatenation
# In such case, we perform noise augmentation first within
# certain SNR rnage on short audio, so the concatenated long
# audio normally contain multiple types of noises (and that's)
# what "multi channel" means here.
stage=0

noise_snr_range="0_10" # The level range of SNR of noise, pos_snr: 10_30
single_speaker=false # Whether we only concatenate wavs from same speaker at p2

segment_length=4

num_bonafides=2580 # Number of bonafide audios to generate
num_spoofs=22800 # Number of spoofed audios to generate

in_data_dir=data/asvspoof2019/LA/dev
p1_data_dir=data/asvspoof2019/LA/mc_p1/dev
p2_data_dir=data/asvspoof2019/LA/mc_p2/dev
p3_data_dir=data/asvspoof2019/LA/mc_p3/dev

. ./kaldi_utils/parse_options.sh

if [ $stage -le 0 ]; then
    echo "$0: Stage 0: Perform pre-processing on the audio, on $in_data_dir"
    python3 pipeline/pre_processing.py \
        --in_data_dir $in_data_dir \
        --out_data_dir $p1_data_dir
fi

if [ $stage -le 1 ]; then
    echo "$0: Stage 1: Perform noise augmentation of the long form audio on $p1_data_dir, with $noise_snr_range dB"
    python3 pipeline/utils/get_utt2dur.py $p1_data_dir

    python3 pipeline/noise_augmentation.py \
        --in_data_dir $p1_data_dir \
        --out_data_dir $p2_data_dir \
	--snr_range "$noise_snr_range"
fi

if [ $stage -le 2 ]; then
    echo "$0: Stage 2: Perform long form concatenation on $p2_data_dir"
    python3 pipeline/utils/get_utt2dur.py $p2_data_dir

    # We can do direct copy because the noise augmentation
    # is performed per-waveform. In such case, we do not
    # get larger amount of data. So the utterance name of
    # the audios do not change.
    cp $in_data_dir/spk2utt $p2_data_dir/spk2utt

    if $single_speaker; then
        single_speaker_flag="--single_speaker"
    else
        single_speaker_flag=""
    fi
    python3 pipeline/long_form_concat.py $single_speaker_flag \
        --num_bonafides $num_bonafides \
        --num_spoofs $num_spoofs \
        --in_data_dir $p2_data_dir \
        --out_data_dir $p3_data_dir
fi

if [ $stage -le 3 ]; then
    echo "$0: Perform segmentation on the long form audio"
    python3 pipeline/long_form_segmentation.py --segment_length $segment_length \
        --in_data_dir $p3_data_dir \
        --out_data_dir $p3_data_dir/SEG$segment_length
fi

if [ $stage -le 4 ]; then
    echo "$0: generate the ultra deepfake CSV file for proper processing"
    python3 pipeline/utils/get_utt2dur.py $p3_data_dir
    python3 pipeline/utils/write_ultra_deepfake_csv.py \
        --in_data_dir $p3_data_dir

    python3 pipeline/utils/get_utt2dur.py $p3_data_dir/SEG$segment_length
    python3 pipeline/utils/write_ultra_deepfake_csv.py \
        --in_data_dir $p3_data_dir/SEG$segment_length
fi
