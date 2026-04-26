#!/usr/bin/env bash
set -euo pipefail

rm -f ~/.claude/foo ~/.claude/.foo.tmp.*
touch ~/.claude/foo
stat -c 'before:%u:%g:%a' ~/.claude/foo

tmp="$(mktemp ~/.claude/.foo.tmp.XXXXXX)"
printf 'edited-via-rename\n' > "$tmp"
mv "$tmp" ~/.claude/foo
stat -c 'after_rename:%u:%g:%a' ~/.claude/foo

python3 - <<'PY'
from pathlib import Path
p = Path.home() / ".claude" / "foo"
p.write_text("edited-via-write\n")
PY
stat -c 'after_write:%u:%g:%a' ~/.claude/foo

if command -v vim >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1; then
  timeout 5s vim -n -u NONE -i NONE -es '+normal Goedited-via-vim' +wq ~/.claude/foo || true
  stat -c 'after_vim:%u:%g:%a' ~/.claude/foo
fi

ls -l ~/.claude/foo
