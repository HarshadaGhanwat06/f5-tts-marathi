from huggingface_hub import hf_hub_download
from pathlib import Path

repo_id = "multilingual-tts/F5-TTS-OpenBible-Marathi"
output_dir = Path("models/openbible-marathi")
output_dir.mkdir(parents=True, exist_ok=True)

files = [
    "model_last.pt",
    "vocab.txt",
    "F5-TTS_OpenBible_Marathi.yaml",
]

for filename in files:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=output_dir,
    )
    print(f"Downloaded: {path}")
