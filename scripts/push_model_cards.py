#!/usr/bin/env python3
"""Push this repo's README.md as the model card of the official HF model repos.

Renders one card per size from README.md:

- prepends HF model-card YAML frontmatter (license, pipeline tag, tags),
- rewrites repo-relative links (DOCS.md, LICENSE, ...) to absolute GitHub
  URLs so they work on huggingface.co,
- adds a link to the GitHub benchmark repo next to the benchmark-site link.

Usage::

    HF_TOKEN=hf_...  python scripts/push_model_cards.py            # push
    python scripts/push_model_cards.py --dry_run --out_dir cards/  # render only

The token needs write access to the nyralabs org.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GITHUB_BLOB = "https://github.com/nyrahealth/CrisperWhisper/blob/main/"
BENCHMARK_REPO = "https://github.com/nyrahealth/nyra_verbatim_speech_benchmark"
SIZES = ("large", "turbo", "medium", "small")

FRONTMATTER = """\
---
language:
  - en
  - de
license: other
license_name: nyra-health-non-commercial-research
license_link: LICENSE.md
pipeline_tag: automatic-speech-recognition
library_name: crisperwhisper
tags:
  - speech-recognition
  - verbatim
  - disfluency
  - whisper
  - ctranslate2
  - word-timestamps
---

"""


def render_card(readme: str) -> str:
    # Repo-relative markdown link targets -> absolute GitHub URLs. Matching on
    # the `](target)` tail (instead of the whole link) also covers nested
    # badge links like [![alt](badge-url)](LICENSE).
    card = re.sub(
        r"\]\((?!https?://|#|mailto:)([^)\s]+)\)",
        rf"]({GITHUB_BLOB}\1)",
        readme,
    )

    # Add the GitHub benchmark repo next to the benchmark-site link in the
    # header link row.
    site_link = "[Benchmark](https://www.nyra-labs.com/research/nyra-verbatim-speech-benchmark)"
    repo_link = f"[Benchmark repo]({BENCHMARK_REPO})"
    if repo_link not in card:
        if site_link in card:
            card = card.replace(site_link, f"{site_link} ·\n{repo_link}", 1)
        else:  # fallback: right under the title
            card = card.replace("\n\n", f"\n\n{repo_link}\n\n", 1)

    return FRONTMATTER + card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", default=list(SIZES), choices=SIZES)
    parser.add_argument("--dry_run", action="store_true", help="Render only, no upload.")
    parser.add_argument("--out_dir", default="", help="With --dry_run: write cards here.")
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create missing repos as private (existing repos keep their visibility).",
    )
    args = parser.parse_args()

    card = render_card((REPO_ROOT / "README.md").read_text(encoding="utf-8"))

    if args.dry_run:
        out_dir = Path(args.out_dir or "model_cards")
        out_dir.mkdir(parents=True, exist_ok=True)
        for size in args.sizes:
            path = out_dir / f"CrisperWhisper2.0_{size}.md"
            path.write_text(card, encoding="utf-8")
            print(f"rendered {path}")
        return

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        sys.exit("HF_TOKEN is required to push (or use --dry_run).")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print("pushing as:", api.whoami()["name"])
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as tmp:
        tmp.write(card)
        card_path = tmp.name
    for size in args.sizes:
        repo_id = f"nyralabs/CrisperWhisper2.0_{size}"
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=args.private)
        api.upload_file(
            path_or_fileobj=card_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Model card: sync README from nyrahealth/CrisperWhisper",
        )
        print(f"pushed model card -> https://huggingface.co/{repo_id}")
    os.unlink(card_path)


if __name__ == "__main__":
    main()
