import os
import time

import torch
import soundfile as sf
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    infer_process,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cuda"

MODEL_DIR = "models/openbible-marathi"

CKPT_PATH = os.path.join(MODEL_DIR, "model_last.pt")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.txt")
CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "F5-TTS_OpenBible_Marathi.yaml",
)

# REF_AUDIO_PATH = "audio/voice1_24k.wav"

# REF_TEXT = "अदितीचा जन्म दोन हजार सतरा साली, आश्विन या महिन्यात, वर्धा इथे झाला."
# REF_TEXT = "सुप्रभात, मी शाळेत जात आहे."
REF_AUDIO_PATH = "audio/hardik.wav"
REF_TEXT = "मी ठीक आहे तुम्ही कसे आहात? मला मदत करा आपण पुन्हा भेटू"
GEN_TEXT = "आपण भारतात राहत असूनही इथल्या वेगवेगळ्या प्रांतात बोलल्या जाणाऱ्या भाषांची माहिती आपल्याला नसते."
OUTPUT_PATH = "outputs/hardik.wav"

# GEN_TEXT = "सुप्रभात, मी शाळेत जात आहे."

# OUTPUT_PATH = "outputs/voice1_24k.wav"
# IMPORTANT:
# Replace this with the EXACT transcript spoken in the
# reference audio.
#REF_TEXT = "आपण भारतात राहत असूनही इथल्या वेगवेगळ्या प्रांतात बोलल्या जाणाऱ्या भाषांची माहिती आपल्याला नसते."

# Marathi text you want to generate.
#GEN_TEXT = "आज आपण कृत्रिम बुद्धिमत्तेच्या नवीन तंत्रज्ञानाबद्दल माहिती घेणार आहोत."

#OUTPUT_PATH = "outputs/marathi_test.wav"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_gpu_memory(stage):
    """Print current GPU memory usage."""

    if not torch.cuda.is_available():
        print(f"[GPU MEMORY] {stage}: CUDA unavailable")
        return

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)

    free, total = torch.cuda.mem_get_info()

    free = free / (1024 ** 3)
    total = total / (1024 ** 3)

    print(f"\n[GPU MEMORY] {stage}")
    print(f"  Allocated : {allocated:.2f} GB")
    print(f"  Reserved  : {reserved:.2f} GB")
    print(f"  Free      : {free:.2f} GB")
    print(f"  Total     : {total:.2f} GB")


def print_separator(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# START
# ============================================================

print_separator("F5-TTS MARATHI INFERENCE")

print("[INFO] Starting inference...")
print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] Model directory: {MODEL_DIR}")
print(f"[INFO] Checkpoint: {CKPT_PATH}")
print(f"[INFO] Vocabulary: {VOCAB_PATH}")
print(f"[INFO] Config: {CONFIG_PATH}")
print(f"[INFO] Reference audio: {REF_AUDIO_PATH}")
print(f"[INFO] Output: {OUTPUT_PATH}")


# ============================================================
# CHECK FILES
# ============================================================

print_separator("CHECKING FILES")

required_files = [
    CKPT_PATH,
    VOCAB_PATH,
    CONFIG_PATH,
    REF_AUDIO_PATH,
]

for path in required_files:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"[ERROR] Required file not found: {path}"
        )

    size_mb = os.path.getsize(path) / (1024 ** 2)

    print(
        f"[OK] {path} "
        f"({size_mb:.2f} MB)"
    )


# ============================================================
# CHECK CUDA
# ============================================================

print_separator("CHECKING GPU")

if not torch.cuda.is_available():
    raise RuntimeError(
        "[ERROR] CUDA is not available. "
        "F5-TTS cannot run on GPU."
    )

gpu_name = torch.cuda.get_device_name(0)

print(f"[OK] CUDA available")
print(f"[OK] GPU: {gpu_name}")
print(f"[OK] PyTorch: {torch.__version__}")
print(f"[OK] CUDA build: {torch.version.cuda}")

print_gpu_memory("Before model loading")


# ============================================================
# LOAD YAML CONFIG
# ============================================================

print_separator("LOADING MODEL CONFIG")

print(f"[INFO] Loading YAML: {CONFIG_PATH}")

cfg = OmegaConf.load(CONFIG_PATH)

model_cfg = cfg.model

print(f"[OK] Model name      : {model_cfg.name}")
print(f"[OK] Tokenizer       : {model_cfg.tokenizer}")
print(f"[OK] Tokenizer path  : {model_cfg.tokenizer_path}")
print(f"[OK] Backbone        : {model_cfg.backbone}")

print("\n[MODEL ARCHITECTURE]")
print(f"  dim              : {model_cfg.arch.dim}")
print(f"  depth            : {model_cfg.arch.depth}")
print(f"  heads            : {model_cfg.arch.heads}")
print(f"  ff_mult          : {model_cfg.arch.ff_mult}")
print(f"  text_dim         : {model_cfg.arch.text_dim}")
print(f"  conv_layers      : {model_cfg.arch.conv_layers}")
print(f"  attn_backend     : {model_cfg.arch.attn_backend}")

print("\n[MEL CONFIGURATION]")
print(f"  sample rate      : {model_cfg.mel_spec.target_sample_rate}")
print(f"  mel channels     : {model_cfg.mel_spec.n_mel_channels}")
print(f"  hop length       : {model_cfg.mel_spec.hop_length}")
print(f"  mel spec type    : {model_cfg.mel_spec.mel_spec_type}")

mel_spec_type = model_cfg.mel_spec.mel_spec_type


# ============================================================
# RESOLVE MODEL CLASS
# ============================================================

print_separator("RESOLVING MODEL CLASS")

model_class_path = (
    f"f5_tts.model.{model_cfg.backbone}"
)

print(f"[INFO] Resolving: {model_class_path}")

model_cls = get_class(model_class_path)

print(f"[OK] Model class: {model_cls}")


# ============================================================
# LOAD VOCODER
# ============================================================

print_separator("LOADING VOCODER")

print(f"[INFO] Vocoder: {mel_spec_type}")
print("[INFO] Device: cuda")
print("[INFO] Loading Vocos...")

vocoder_start = time.time()

vocoder = load_vocoder(
    vocoder_name=mel_spec_type,
    device=DEVICE,
)

vocoder_time = time.time() - vocoder_start

print(
    f"[OK] Vocos loaded in {vocoder_time:.2f} seconds"
)

print_gpu_memory("After vocoder loading")


# ============================================================
# LOAD F5-TTS MODEL
# ============================================================

print_separator("LOADING F5-TTS CHECKPOINT")

print(f"[INFO] Checkpoint: {CKPT_PATH}")
print("[INFO] use_ema=True")
print("[INFO] ODE method: euler")
print("[INFO] Loading model onto CUDA...")
print("[INFO] This may take some time...")

model_start = time.time()

model = load_model(
    model_cls,
    model_cfg.arch,
    CKPT_PATH,
    mel_spec_type=mel_spec_type,
    vocab_file=VOCAB_PATH,
    ode_method="euler",
    use_ema=True,
    device=DEVICE,
)

model.eval()

model_time = time.time() - model_start

print(
    f"[OK] F5-TTS model loaded in {model_time:.2f} seconds"
)

print_gpu_memory("After F5-TTS model loading")


# ============================================================
# CHECK REFERENCE TEXT
# ============================================================

print_separator("CHECKING REFERENCE TEXT")

if (
    not REF_TEXT
    or REF_TEXT.strip() == ""
    or REF_TEXT.strip()
    == "YOUR EXACT MARATHI REFERENCE TRANSCRIPT HERE"
):
    raise ValueError(
        "\n[ERROR] REF_TEXT is still a placeholder.\n"
        "You MUST replace REF_TEXT with the exact "
        "Marathi transcript spoken in the reference audio."
    )

print("[OK] Reference transcript provided")
print(f"[REF TEXT] {REF_TEXT}")


# ============================================================
# CHECK GENERATION TEXT
# ============================================================

print_separator("GENERATION TEXT")

if not GEN_TEXT.strip():
    raise ValueError(
        "[ERROR] GEN_TEXT cannot be empty."
    )

print(f"[GEN TEXT] {GEN_TEXT}")


# ============================================================
# PREPROCESS REFERENCE AUDIO
# ============================================================

print_separator("PROCESSING REFERENCE AUDIO")

print(f"[INFO] Audio: {REF_AUDIO_PATH}")
print("[INFO] Processing reference audio + transcript...")

ref_start = time.time()

ref_audio, ref_text = preprocess_ref_audio_text(
    REF_AUDIO_PATH,
    REF_TEXT,
)

ref_time = time.time() - ref_start

print(
    f"[OK] Reference preprocessing completed "
    f"in {ref_time:.2f} seconds"
)

print(f"[REF TEXT USED] {ref_text}")


# ============================================================
# RUN INFERENCE
# ============================================================

print_separator("RUNNING F5-TTS INFERENCE")

print("[INFO] Starting speech generation...")
print(f"[INFO] Target text: {GEN_TEXT}")
print("[INFO] NFE steps: 32")
print("[INFO] CFG strength: 2.0")
print("[INFO] Speed: 1.0")
print("[INFO] Device: CUDA")

print_gpu_memory("Before inference")

torch.cuda.synchronize()

inference_start = time.time()

with torch.inference_mode():

    final_wave, sample_rate, spectrogram = infer_process(
        ref_audio,
        ref_text,
        GEN_TEXT,
        model,
        vocoder,
        mel_spec_type=mel_spec_type,
        nfe_step=32,
        cfg_strength=2.0,
        sway_sampling_coef=-1.0,
        speed=1.0,
        device=DEVICE,
    )

torch.cuda.synchronize()

inference_time = time.time() - inference_start

print(
    f"\n[OK] Inference completed "
    f"in {inference_time:.2f} seconds"
)

print_gpu_memory("After inference")


# ============================================================
# VALIDATE OUTPUT
# ============================================================

print_separator("VALIDATING OUTPUT")

if final_wave is None:
    raise RuntimeError(
        "[ERROR] F5-TTS returned no waveform."
    )

print("[OK] Waveform generated")

print(f"[INFO] Sample rate: {sample_rate}")

try:
    waveform_length = len(final_wave)
    duration = waveform_length / sample_rate

    print(f"[INFO] Samples: {waveform_length}")
    print(f"[INFO] Duration: {duration:.2f} seconds")

except Exception:
    print("[WARNING] Could not calculate waveform duration.")


# ============================================================
# SAVE WAV
# ============================================================

print_separator("SAVING AUDIO")

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True,
)

print(f"[INFO] Saving to: {OUTPUT_PATH}")

sf.write(
    OUTPUT_PATH,
    final_wave,
    sample_rate,
)

if os.path.isfile(OUTPUT_PATH):

    output_size = (
        os.path.getsize(OUTPUT_PATH)
        / (1024 ** 2)
    )

    print(
        f"[OK] Audio saved successfully "
        f"({output_size:.2f} MB)"
    )

else:
    raise RuntimeError(
        "[ERROR] Output WAV was not created."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print_separator("INFERENCE COMPLETE")

print(f"[SUCCESS] Output file : {OUTPUT_PATH}")
print(f"[SUCCESS] Sample rate  : {sample_rate} Hz")
print(f"[SUCCESS] Duration     : {duration:.2f} sec")
print(f"[SUCCESS] Inference    : {inference_time:.2f} sec")

print_gpu_memory("Final")

print("\n[INFO] You can now copy the WAV back to your local machine.")
print("=" * 70)
