"""Set the published sdist source in the conda recipe for a tagged release."""

import argparse
import re
from pathlib import Path

SOURCE_PATTERN = re.compile(r"(?ms)^source:\n(?P<body>.*?)(?=^build:|\Z)")
VALID_SOURCE = re.compile(
    r"(?:  #[^\n]*\n)*(?:  path: \.\.|  url: [^\n]+\n  sha256: [0-9a-fA-F]{64})"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("sha256")
    parser.add_argument(
        "recipe", type=Path, nargs="?", default=Path(__file__).with_name("meta.yaml")
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+[A-Za-z0-9.+_-]*", args.version):
        parser.error("version must be a release version such as 0.14.0")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", args.sha256):
        parser.error("sha256 must contain 64 hexadecimal characters")

    source = (
        "source:\n"
        "  url: https://pypi.io/packages/source/s/sklearn-genetic-opt/"
        f"sklearn_genetic_opt-{args.version}.tar.gz\n"
        f"  sha256: {args.sha256}"
    )
    recipe = args.recipe.read_text(encoding="utf-8")
    matches = list(SOURCE_PATTERN.finditer(recipe))
    if len(matches) != 1:
        parser.error("Unsupported conda recipe source; expected one local path or sdist URL")
    match = matches[0]
    body = match.group("body")
    content = body.rstrip("\n")
    if VALID_SOURCE.fullmatch(content) is None:
        parser.error("Unsupported conda recipe source; expected one local path or sdist URL")
    updated = recipe[: match.start()] + source + body[len(content) :] + recipe[match.end() :]
    if updated != recipe:
        args.recipe.write_text(updated, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
