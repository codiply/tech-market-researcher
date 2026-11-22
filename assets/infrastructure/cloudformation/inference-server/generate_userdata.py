#!/usr/bin/env python3
"""Regenerate the compressed setup-script blob in inference-server.yaml.

EC2 limits user-data to 16 KiB of base64, which the plaintext setup script
exceeds. The source of truth is scripts/inference-setup.sh; this tool
strips the bash comment lines (heredoc contents are kept), gzips and
base64-encodes the result, and rewrites the blob in the template's
user-data trampoline. Run it after editing the script, before deploying
the stack:

    python3 generate_userdata.py
"""

import base64
import gzip
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "inference-server.yaml"
SCRIPT = HERE / "scripts" / "inference-setup.sh"
BLOB_RE = re.compile(r'echo "[^"]*" \| base64 -d \| gunzip -c > /var/lib/inference-setup/setup\.sh')
USERDATA_LIMIT_B64 = 16384


def strip_comments(text: str) -> str:
    """Drop bash comment lines so the compressed blob fits the user-data limit.

    Comments are only stripped outside heredocs: their contents (shebangs,
    Python/JS comments, systemd unit comments) are shipped as-is, and the
    repo keeps the full commented script as the source of truth.
    """
    out, in_heredoc, delim = [], False, None
    for line in text.splitlines(keepends=True):
        if in_heredoc:
            out.append(line)
            if line.strip() == delim:
                in_heredoc = False
            continue
        m = re.search(r"<<-?\s*['\"]?([A-Za-z_]+)['\"]?\s*$", line)
        if m:
            delim = m.group(1)
            in_heredoc = True
        if line.lstrip().startswith("#") and not line.lstrip().startswith("#!"):
            continue
        out.append(line)
    return "".join(out)


def _child(node: yaml.Node, key: str) -> yaml.Node:
    """Return the child node of a mapping node with the given key."""
    for k, v in node.value:
        if k.value == key:
            return v
    raise KeyError(key)


def check_limit(template_text: str) -> int:
    """Return the final user-data base64 size, failing if it exceeds the EC2 limit."""
    doc = yaml.compose(template_text)
    node = doc
    for key in (
        "Resources",
        "InferenceServerLaunchTemplate",
        "Properties",
        "LaunchTemplateData",
        "UserData",
        "Fn::Base64",
        "Fn::Join",
    ):
        node = _child(node, key)
    parts = [p.value for p in node.value[1].value]
    assert all(isinstance(p, str) for p in parts), "unexpected non-text user-data part"
    size = len(base64.b64encode("".join(parts).encode()))
    if size > USERDATA_LIMIT_B64:
        sys.exit(f"error: user-data is {size} bytes of base64, over the EC2 limit of {USERDATA_LIMIT_B64}")
    return size


def main() -> None:
    """Regenerate the compressed blob in the template and check the user-data size."""
    template = TEMPLATE.read_text()
    script = strip_comments(SCRIPT.read_text())
    blob = base64.b64encode(gzip.compress(script.encode(), 9)).decode()
    if not BLOB_RE.search(template):
        sys.exit(f"error: could not find the compressed setup-script blob in {TEMPLATE}")
    blob_line = f'echo "{blob}" | base64 -d | gunzip -c > /var/lib/inference-setup/setup.sh'
    template = BLOB_RE.sub(lambda _: blob_line, template, count=1)
    size = check_limit(template)
    TEMPLATE.write_text(template)
    print(f"blob written: {len(blob)} bytes of base64 (user-data total {size} / {USERDATA_LIMIT_B64})")


if __name__ == "__main__":
    main()
