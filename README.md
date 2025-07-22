# LENS-DF-DataGen

[![Sampledata](https://img.shields.io/badge/Sample%20data-Zenodo-9cf?logo=zenodo)](https://zenodo.org/records/15948624)
[![Conference](https://img.shields.io/badge/Conference-IJCB%202025-green)](https://ijcb2025.ieee-biometrics.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](https://zenodo.org/records/15948624/files/LICENSE?download=1)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2507.08530-b31b1b.svg)](https://arxiv.org/abs/2507.08530) -->

(This work has been accepted to [IEEE IJCB 2025](https://ijcb2025.ieee-biometrics.org). The arxiv link of the camera-ready paper will be added later)

This repository provides the data generation recipe for LENS-DF, designed to create long-form, multi-speaker, and noisy audio data. 
This repo is useful for developing robust audio deepfake detection systems and performing temporal localization.

## Getting Started

### Dependencies
You can install the Python dependencies using anaconda. Below is an example:
```bash
conda create --name lensdf python==3.9.0
conda activate lensdf
conda install pip==24.0

pip install -r requirements.txt
```
Additionally, you'll need the sv56 toolkit for volume normalization,
which can be found at [here](https://github.com/nii-yamagishilab/SSL-SAS/tree/38218718e512468dd623e944ea8a50c1f8400625/scripts). 
See `install.sh` for downloading it, and `sub_sv56.sh` for usage.


### Data preparation
To get started, we recommend organzing the source directory as follows:

We recommend to prepare the source directory with below file architecture:
```
in_data_dir/
├── ultra_deepfake.csv
├── data.csv
├── wavs/
│   ├── wav1.wav
│   ├── wav2.wav
│   └── ...
...
```
Place all your audio files in the wavs/ folder. 
Your data.csv should list these files and their corresponding metadata, like in this example:
```
,file,label,speaker,attack
0,source_data/wavs/wav1.wav,bonafide,spk1,-
1,source_data/wavs/wav2.wav,bonafide,spk2,-
2,source_data/wavs/wav3.wav,spoof,spk1,vc1
...
```
NOTE: Since we are not sure about the label and attack assignment scheme from user's side, please prepare data.csv by yourself.

### Data generation
As an example, please refer to `scripts/` for more details.
There are two scripts in the repository, `single_channel.sh` and `multi_channel.sh`.
While they appear similar from a caller's perspective, `single_channel.sh` generates long-form waves where each wave only has one type of noise.

At the beginning of both bash scripts, you'll find a configuration block like this:
```
in_data_dir=data/asvspoof2019/LA/dev
p1_data_dir=data/asvspoof2019/LA/mc_p1/dev
p2_data_dir=data/asvspoof2019/LA/mc_p2/dev
p3_data_dir=data/asvspoof2019/LA/mc_p3/dev
```
Here, you can specify your input data directory and the output directories for each stage of the generation process. 
This allows you to easily access and examine the intermediate data generated at each stage, as covered in the LENS-DF paper.

The resulting `$p3_data_dir` will expectedly have the following file structure:
```
p3_data_dir/
├── ultra_deepfake.csv
├── data.csv
├── wavs/
│   ├── wav1.wav
│   ├── wav2.wav
│   └── ...
└── SEG_N/
    ├── ultra_deepfake.csv
    └── wavs/
        ├── wav1_1.wav
        ├── wav1_2.wav
        └── ...
```

### Audio DeepFake detection
For conducting experiments such as training audio DeepFake detectors and benchmarking, please refer to [Anti-DeepFake](https://github.com/nii-yamagishilab/AntiDeepfake) for more details. 

After the final generation step, you'll find an `ultra_deepfake.csv` file. 
This CSV contains all the necessary metadata for the generated waveforms and serves as the primary input for training and evaluating your models.


## Important notes
- Computational Resources: While the data generation process itself doesn't require a GPU, the overall generation time can vary significantly depending on the amount of original and generated data due to the sequential nature of the process.
- Source Data: Some generation recipes might depend on pre-existing source datasets (e.g., ADD, ASVspoof 2019). Please consult the specific script's documentation or the LENS-DF paper for details on how to obtain these.
- Reproducibility: Minor variations in results might occur due to differences in system environments or library versions.

## License
This project is licensed under the MIT license. See LICENSE for details.

## Acknowledgements
This study is supported by the New Energy and Industrial Technology Development Organization (NEDO, JPNP22007), JST AIP Acceleration Research (JPMJCR24U3), and JST PRESTO (JPMJPR23P9). This study was partially carried out using the TSUBAME4.0 supercomputer at the Institute of Science Tokyo.

## Citation
If you find this work interesting and useful for your research, please consider citing our paper:
```
@inproproceedings{Liu2025LENSDF,
  author = {Liu, Xuechen and Ge, Wanying and Wang, Xin and Yamagishi, Junichi},
  title = {LENS-DF: Deepfake Detection and Temporal Localization for Long-Form Noisy Speech},
  booktitle = {IEEE International Joint Conference on Biometrics (IJCB)},
  address = {Osaka, Japan},
  year = {2025},
}
```