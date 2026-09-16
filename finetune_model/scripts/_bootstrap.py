"""Path setup shared by every script: finetune_model/src plus the repo's model/ package."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)                       # finetune_model/
REPO = os.path.dirname(ROOT)                        # repo root
for p in (os.path.join(ROOT, "src"), os.path.join(REPO, "model")):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.stdout.reconfigure(line_buffering=True)         # nohup logs stay readable live
