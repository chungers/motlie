#!/usr/bin/env bash
# Pull a known-size blob and verify checksum. Catches MTU truncation, mid-stream
# resets, and userspace egress packet-reassembly bugs.
set -u
TEST_NAME=vnet-08-mtu-large-transfer
declare -F pass >/dev/null 2>&1 || . "$(dirname "$0")/../shared/result.sh"

if ! command -v curl >/dev/null 2>&1 || ! command -v sha256sum >/dev/null 2>&1; then
  skip "missing-tool"
fi
# A small, stable, well-known blob with a published sha256. We use the GNU
# coreutils tarball mirror redirect; if this changes, pin a different artifact.
url="https://ftp.gnu.org/gnu/hello/hello-2.10.tar.gz"
expected_sha256="31e066137a962676e89f69d1b65382de95a7ef7d914b8cb956f41ea72e0f516b"
out=/tmp/${TEST_NAME}.bin
if ! curl -fsSL --max-time 30 -o "$out" "$url"; then
  rm -f "$out"
  fail "download-failed url=$url"
fi
got=$(sha256sum "$out" | awk '{print $1}')
size=$(stat -c '%s' "$out")
rm -f "$out"
if [[ "$got" != "$expected_sha256" ]]; then
  fail "sha-mismatch size=$size got=$got expected=$expected_sha256"
fi
pass "size=$size sha=ok"
