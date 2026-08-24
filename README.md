# Marathi F5-TTS

Fine-tuning and inference pipeline for Marathi Text-to-Speech using
[F5-TTS](https://github.com/SWivid/F5-TTS).

## Overview

This repository contains the Marathi F5-TTS setup, dataset preparation,
vocabulary validation, fine-tuning and inference workflow.

The project includes:

-   Marathi OpenBible F5-TTS as the pretrained Marathi base model
-   RASA Marathi female/emotion experiment metadata
-   IISc SYSPIN Marathi Female dataset preparation
-   Custom Marathi vocabulary
-   F5-TTS fine-tuning
-   Training checkpoint/sample generation
-   Direct Marathi inference
-   Gradio inference using a custom fine-tuned checkpoint

> Large datasets, model checkpoints, generated audio, logs and the
> Python virtual environment are excluded from Git because of their
> size.

## Repository Structure

``` text
f5-tts-marathi/
├── datasets/
│   └── syspin_marathi_female/
│       ├── extracted/                  # Raw audio - ignored
│       ├── audit_syspin.py
│       ├── check_syspin_oov.py
│       ├── metadata_all.csv
│       ├── metadata_5h.csv
│       ├── prepare_syspin.py
│       ├── syspin_characters.txt
│       ├── missing_wavs.txt
│       └── wav_without_transcript.txt
├── models/
│   └── openbible-marathi/
│       ├── F5-TTS_OpenBible_Marathi.yaml
│       ├── vocab.txt
│       └── model_last.pt              # ignored; restore separately
├── rasa_female_emotions/
│   └── rasa_train_metadata.csv
├── audio/                             # ignored
├── outputs/                           # ignored
├── logs/                              # ignored
├── f5tts/                             # virtual environment; ignored
├── check_marathi_vocab.py
├── check_syspin_oov.py
├── download_model.py
├── download_one_marathi_voice.py
├── infer_marathi.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

Recommended:

-   Ubuntu/Linux
-   Python 3.12
-   NVIDIA GPU
-   CUDA-compatible PyTorch
-   Git

Training was performed on an NVIDIA RTX PRO 6000 Blackwell GPU with
approximately 96 GB VRAM.

## Installation

### 1. Clone

``` bash
git clone https://github.com/HarshadaGhanwat06/f5-tts-marathi.git
cd f5-tts-marathi
```

### 2. Create and activate the environment

``` bash
python3.12 -m venv f5tts
source f5tts/bin/activate
python --version
```

### 3. Install dependencies

``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify PyTorch/CUDA

``` bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
nvidia-smi
```

`torch.cuda.is_available()` should return `True` on a correctly
configured GPU machine.

## Base Marathi Model

The fine-tuning experiments start from the Marathi OpenBible F5-TTS
checkpoint.

Required:

``` text
models/openbible-marathi/
├── model_last.pt
├── vocab.txt
└── F5-TTS_OpenBible_Marathi.yaml
```

`model_last.pt` is several GB and is not stored in Git. Restore/download
it separately and place it at:

``` text
models/openbible-marathi/model_last.pt
```

The repository tracks the Marathi vocabulary and model configuration.

## Marathi Vocabulary

The custom vocabulary is:

``` text
models/openbible-marathi/vocab.txt
```

The training/inference tokenizer is configured as:

``` text
--tokenizer custom
--tokenizer_path models/openbible-marathi/vocab.txt
```

Check the vocabulary with:

``` bash
python check_marathi_vocab.py
```

## Datasets

### RASA Marathi Female

RASA Marathi female/emotion data was used for Marathi female voice and
emotion experiments.

Metadata:

``` text
rasa_female_emotions/rasa_train_metadata.csv
```

The experiment was used to investigate female voice characteristics,
emotional speech, reference voice similarity and prosody.

Original audio is not included in this repository.

### IISc SYSPIN Marathi Female

The main fine-tuning experiment used the IISc SYSPIN Marathi Female
corpus.

Dataset audit:

``` text
Transcript entries : 21,790
WAV files          : 21,790
Missing WAVs       : 0
WAV without text   : 0
Empty transcripts  : 0
Total duration     : 51.06 hours
Mean duration      : 8.44 sec
Minimum duration   : 0.51 sec
Maximum duration   : 24.10 sec
Sample rate        : 48000 Hz
Channels           : 1
Sample width       : 3 bytes
```

The selected speaker is a Marathi female speaker. The corpus covers
domains such as books, weather, agriculture, health, finance, politics,
education, general content and evaluation.

Raw extracted audio is approximately 25 GB and is excluded from Git.

## SYSPIN Preparation

Place/extract the SYSPIN audio/transcript data under:

``` text
datasets/syspin_marathi_female/extracted/
```

### Audit

``` bash
python datasets/syspin_marathi_female/audit_syspin.py
```

The audit checks transcript/audio matching, duration, sample rate,
channels, sample width and character coverage.

A valid dataset should have:

``` text
Missing WAVs            : 0
WAV without transcript : 0
Empty transcripts       : 0
```

### OOV Check

``` bash
python datasets/syspin_marathi_female/check_syspin_oov.py
```

The original SYSPIN transcripts contained unsupported characters. The
preparation pipeline normalized/removed unsupported characters so the
final training text had:

``` text
OOV characters : 0
```

Do not start fine-tuning until the training text is compatible with the
custom vocabulary.

### Prepare Metadata

``` bash
python datasets/syspin_marathi_female/prepare_syspin.py
```

The project contains:

``` text
metadata_all.csv   # full corpus
metadata_5h.csv    # selected 5-hour subset
```

The selected subset contained:

``` text
2128 audio files
5.00 hours
```

Metadata uses the F5-TTS format:

``` text
audio_file|text
```

## Prepare Dataset for F5-TTS

For the 5-hour SYSPIN subset:

``` bash
python f5tts/lib/python3.12/site-packages/f5_tts/train/datasets/prepare_csv_wavs.py     datasets/syspin_marathi_female/metadata_5h.csv     f5tts/data/SYSPIN_Marathi_Female_5h     --workers 32
```

The prepared dataset is created under:

``` text
f5tts/data/SYSPIN_Marathi_Female_5h/
```

## Fine-Tuning

This project performs **fine-tuning of an existing Marathi F5-TTS
model**, not training F5-TTS from scratch.

Pipeline:

``` text
Marathi OpenBible F5-TTS
          +
SYSPIN Marathi Female
          +
Custom Marathi vocabulary
          |
          v
Fine-tuned Marathi Female F5-TTS
```

### Main Configuration

``` text
Experiment       : F5TTS_v1_Base
Dataset          : SYSPIN_Marathi_Female_5h
Tokenizer        : custom
Batch size       : 4 samples/GPU
Batch type       : sample
Learning rate    : 1e-5
Warmup updates   : 100
Checkpoint save  : every 200 updates
Last checkpoint  : every 100 updates
Keep checkpoints : 2
Final experiment: 40 epochs
```

### Start Training

``` bash
source f5tts/bin/activate

accelerate launch f5tts/lib/python3.12/site-packages/f5_tts/train/finetune_cli.py     --exp_name F5TTS_v1_Base     --dataset_name SYSPIN_Marathi_Female_5h     --finetune     --pretrain /root/f5-tts-marathi/models/openbible-marathi/model_last.pt     --tokenizer custom     --tokenizer_path /root/f5-tts-marathi/models/openbible-marathi/vocab.txt     --batch_size_per_gpu 4     --batch_size_type sample     --epochs 40     --num_warmup_updates 100     --save_per_updates 200     --last_per_updates 100     --keep_last_n_checkpoints 2     --learning_rate 1e-5     --log_samples
```

Replace `/root/f5-tts-marathi` with the actual repository path if
different.

### Background Training

``` bash
nohup accelerate launch f5tts/lib/python3.12/site-packages/f5_tts/train/finetune_cli.py     --exp_name F5TTS_v1_Base     --dataset_name SYSPIN_Marathi_Female_5h     --finetune     --pretrain /root/f5-tts-marathi/models/openbible-marathi/model_last.pt     --tokenizer custom     --tokenizer_path /root/f5-tts-marathi/models/openbible-marathi/vocab.txt     --batch_size_per_gpu 4     --batch_size_type sample     --epochs 40     --num_warmup_updates 100     --save_per_updates 200     --last_per_updates 100     --keep_last_n_checkpoints 2     --learning_rate 1e-5     --log_samples     > training.log 2>&1 &
```

Monitor training:

``` bash
tail -f training.log
```

Check the process:

``` bash
ps aux | grep finetune_cli
```

Check GPU:

``` bash
nvidia-smi
```

## Checkpoints and Training Samples

Fine-tuned checkpoints are stored under:

``` text
f5tts/lib/python3.12/ckpts/SYSPIN_Marathi_Female_5h/
```

Typical files:

``` text
model_last.pt
model_XXXXX.pt
pretrained_model_last.pt
```

With `--log_samples`, generated/reference samples are stored under:

``` text
f5tts/lib/python3.12/ckpts/SYSPIN_Marathi_Female_5h/samples/
```

For example:

``` text
update_255200_ref.wav
update_255200_gen.wav
update_255400_ref.wav
update_255400_gen.wav
update_255600_ref.wav
update_255600_gen.wav
```

`ref.wav` is the reference recording and `gen.wav` is generated from the
checkpoint.

## Training Result

The final 40-epoch experiment completed at:

``` text
Epoch 39/40 : loss = 0.940
Epoch 40/40 : loss = 0.462
Final update: 21280
```

The final checkpoint was:

``` text
f5tts/lib/python3.12/ckpts/SYSPIN_Marathi_Female_5h/model_last.pt
```

Loss decreased substantially, but TTS quality was not judged from loss
alone.

Observed generated-audio issues included:

-   Male/female voice mixing
-   Inconsistent speaker identity
-   Incorrect pronunciation in some generations
-   Unwanted angry/frustrated prosody
-   Emotion not consistently matching the generation text

Therefore, checkpoint audio evaluation is required in addition to loss
monitoring.

# Gradio Inference

F5-TTS provides a Gradio UI for interactive inference.

Start the environment:

``` bash
source f5tts/bin/activate
```

Launch Gradio:

``` bash
f5-tts_infer-gradio --host 0.0.0.0 --port 7860
```

Open:

``` text
http://SERVER_IP:7860
```

For a remote GPU server, make sure port `7860` is reachable.

## Use the Fine-Tuned Marathi Checkpoint

In the Gradio UI select:

``` text
Choose TTS Model → Custom
```

Set the three custom fields to:

### Model

``` text
/root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/SYSPIN_Marathi_Female_5h/model_last.pt
```

### Vocabulary

``` text
/root/f5-tts-marathi/models/openbible-marathi/vocab.txt
```

### Config

``` text
/root/f5-tts-marathi/models/openbible-marathi/F5-TTS_OpenBible_Marathi.yaml
```

These must be compatible with the installed F5-TTS implementation.

## Reference Audio

Upload a clean Marathi female reference recording.

Prefer:

-   One speaker
-   Low background noise
-   No music
-   Clear speech
-   Short recording
-   Accurate transcript

Enter the exact text spoken in the reference audio.

Example:

``` text
आम्ही कोणतीही चोरी केली नाही.
```

## Generation Text

Enter the Marathi text to synthesize.

Example:

``` text
आज आपण नवीन प्रकल्पावर काम करणार आहोत.
```

The inference flow is:

``` text
Reference Audio
      +
Reference Text
      +
Generation Text
      +
Fine-tuned Marathi F5-TTS
      |
      v
Generated Marathi Audio
```

## Gradio Parameters

Recommended starting values:

``` text
NFE Steps           : 32
Speed               : 1
Cross-Fade Duration : 0.15
```

Use a fixed seed when comparing different checkpoints.

## Direct Inference

The repository also contains:

``` text
infer_marathi.py
```

Configure the checkpoint, vocabulary, model config, reference audio,
reference text and generation text in the script, then run:

``` bash
python infer_marathi.py
```

## Checkpoint Evaluation

Compare generated samples from different updates:

``` text
update_XXXX_gen.wav
```

Evaluate:

1.  Female voice consistency
2.  Speaker similarity
3.  Marathi pronunciation
4.  Naturalness
5.  Prosody
6.  Artifacts
7.  Stability
8.  Emotion

Do not select a checkpoint only because it has the lowest training loss.

# Troubleshooting

## Gradio generation fails

Check the Gradio terminal output and verify:

``` text
Checkpoint path
Vocabulary path
Model config path
```

Also verify that the checkpoint and configuration are compatible with
the installed F5-TTS version.

## `DiT.__init__() got an unexpected keyword argument 'hydra'`

This indicates a mismatch between the model configuration and the
installed F5-TTS implementation.

Use a configuration compatible with the installed `DiT` implementation.
The checkpoint, config and installed F5-TTS version must match.

## Generated voice sounds male/female mixed

Possible causes include:

-   Base model speaker characteristics
-   Insufficient speaker adaptation
-   Reference audio conditioning
-   Fine-tuning configuration
-   Checkpoint selection
-   Training instability
-   Dataset characteristics

Compare multiple checkpoints using the same reference audio and text.

## Marathi pronunciation is incorrect

Run:

``` bash
python check_marathi_vocab.py
python datasets/syspin_marathi_female/check_syspin_oov.py
```

Make sure the generation text uses characters supported by the custom
vocabulary.

## Monitor GPU

``` bash
nvidia-smi
```

Check training:

``` bash
ps aux | grep finetune_cli
```

## Monitor logs

``` bash
tail -f training.log
```

## Disk Usage

``` bash
du -sh * | sort -hr
df -h
```

Large local directories include:

``` text
f5tts/
datasets/.../extracted/
models/.../model_last.pt
f5tts/.../ckpts/
logs/
```

These are excluded from Git.

# Git / Large Files

The repository intentionally does not track:

``` text
f5tts/
datasets/**/extracted/
*.pt
*.pth
*.ckpt
*.safetensors
*.wav
*.mp3
*.flac
logs/
outputs/
audio/
__pycache__/
```

The repository tracks source code, dataset metadata, vocabulary, model
configuration, requirements and documentation.

# Reproduce the SYSPIN Experiment

``` bash
git clone https://github.com/HarshadaGhanwat06/f5-tts-marathi.git
cd f5-tts-marathi

python3.12 -m venv f5tts
source f5tts/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

nvidia-smi

python -c "import torch; print(torch.cuda.is_available())"

python check_marathi_vocab.py

python datasets/syspin_marathi_female/audit_syspin.py

python datasets/syspin_marathi_female/check_syspin_oov.py

python datasets/syspin_marathi_female/prepare_syspin.py

python f5tts/lib/python3.12/site-packages/f5_tts/train/datasets/prepare_csv_wavs.py     datasets/syspin_marathi_female/metadata_5h.csv     f5tts/data/SYSPIN_Marathi_Female_5h     --workers 32
```

Then start fine-tuning using the command above.

After training:

``` bash
f5-tts_infer-gradio --host 0.0.0.0 --port 7860
```

# Future Work

Potential improvements include:

-   Better female speaker consistency
-   Additional Marathi female speakers
-   Better speaker balancing
-   More emotion-balanced Marathi data
-   Separate emotion-specific fine-tuning
-   Better checkpoint selection
-   More Marathi pronunciation coverage
-   Learning-rate and epoch experiments
-   Objective TTS evaluation
-   Emotion-specific evaluation for happy, sad, angry, fear and disgust
-   Comparison of RASA and SYSPIN models

# References

-   [F5-TTS](https://github.com/SWivid/F5-TTS)
-   IISc SYSPIN corpus and associated dataset resources
-   Marathi OpenBible F5-TTS model used as the pretrained Marathi base

# Author

**Harshada Ghanwat**

GitHub: https://github.com/HarshadaGhanwat06

Repository: https://github.com/HarshadaGhanwat06/f5-tts-marathi
