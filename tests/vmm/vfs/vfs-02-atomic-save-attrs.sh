#!/usr/bin/env bash
# `sed -i` performs a tmpfile-write + rename. Assert the result file keeps
# the original uid/gid/mode (the vi/atomic-save pattern).
set -u
TEST_NAME=vfs-02-atomic-save-attrs
declare -F pass >/dev/null 2>&1 || . "$(dirname "$0")/../shared/result.sh"

f=/workspace/.vmm-atomic-$$
echo "alpha" > "$f"
chmod 0640 "$f"
before=$(stat -c '%u:%g:%a' "$f")
sed -i -e 's/alpha/beta/' "$f"
after=$(stat -c '%u:%g:%a' "$f")
content=$(cat "$f")
rm -f "$f"
if [[ "$content" != "beta" ]]; then
  fail "content-wrong got='$content'"
fi
if [[ "$before" != "$after" ]]; then
  fail "attrs-changed before=$before after=$after"
fi
pass "$after"
