#!/usr/bin/env python3
"""Download demo BounceBRDF data from Hugging Face into the data/ directory."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "fau-vce/BounceBRDF"
SUBSETS = ["arch1", "re2", "vr1", "vr3"]
DATA_DIR = Path(__file__).parent.parent.parent / "data"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "subset",
        nargs="?",
        choices=SUBSETS,
        metavar=f"{{{','.join(SUBSETS)}}}",
        help="Download only this subset. Omit to download everything.",
    )
    args = parser.parse_args()

    allow_patterns = [f"{args.subset}/*", "README.md"] if args.subset else None

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = f"'{args.subset}'" if args.subset else "full dataset"
    print(f"Downloading {target} from '{REPO_ID}' into '{DATA_DIR}' ...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(DATA_DIR),
        allow_patterns=allow_patterns,
    )
    print(f"Done. Data saved to: {DATA_DIR}")
