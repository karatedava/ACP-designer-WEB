from pathlib import Path
import os
import sys
from huggingface_hub import snapshot_download

# ────────────────────────────────────────────────
MODEL_REPO  = "karatedava/PeogenACP"
MODEL_LOCAL = "src/models/generative/progen2-ACP-inference"
# ────────────────────────────────────────────────

local_path = Path(MODEL_LOCAL)

if not local_path.exists() or not any(local_path.iterdir()):
    print(f"Model directory not found or empty → downloading {MODEL_REPO} ...")
    snapshot_download(
        repo_id          = MODEL_REPO,
        local_dir        = str(local_path),
        local_dir_use_symlinks = False,
        # token          = os.getenv("HF_TOKEN"),     # uncomment if gated/private
        # allow_patterns = ["*.json", "*.safetensors", "*.bin"],  # optional
        # ignore_patterns = ["*.msgpack", "*.h5"],               # optional
    )
    print("Download finished\n")
else:
    print(f"Model already present at {MODEL_LOCAL}\n")

print("Starting application ...")
# os.chdir(str(local_path.parent))           # optional: set cwd to /src/models/generative

# Replace current process with app.py (best signal handling)
os.execv(
    sys.executable,
    [sys.executable, "app.py"] + sys.argv[1:]
)