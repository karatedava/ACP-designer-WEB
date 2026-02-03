from pathlib import Path
import os
import sys
from huggingface_hub import snapshot_download

# ────────────────────────────────────────────────
MODEL_REPO_ACP  = "karatedava/ProGenACP"
MODEL_REPO_MP  = "karatedava/ProGenMP"

MODEL_LOCAL_ACP = "src/models/generative/progen2-ACP-inference"
MODEL_LOCAL_MP = "src/models/generative/progen2-MP-inference"
# ────────────────────────────────────────────────

def prepare_model(repo:str, local:str):

    local_path = Path(local)

    if not local_path.exists() or not any(local_path.iterdir()):
        print(f"Model directory not found or empty → downloading {repo} ...")
        snapshot_download(
            repo_id          = repo,
            local_dir        = str(local_path),
            local_dir_use_symlinks = False,
            # token          = os.getenv("HF_TOKEN"),     # uncomment if gated/private
            # allow_patterns = ["*.json", "*.safetensors", "*.bin"],  # optional
            # ignore_patterns = ["*.msgpack", "*.h5"],               # optional
        )
        print("Download finished\n")
    else:
        print(f"Model already present at {local_path}\n")

    print("Starting application ...")
    # os.chdir(str(local_path.parent))           # optional: set cwd to /src/models/generative

prepare_model(MODEL_REPO_ACP, MODEL_LOCAL_ACP)
prepare_model(MODEL_REPO_MP, MODEL_LOCAL_MP)

# Replace current process with app.py (best signal handling)
os.execv(
    sys.executable,
    [sys.executable, "app.py"] + sys.argv[1:]
)