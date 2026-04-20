#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"
BASE_VM_NAME="${MOTLIE_VZ_BASE_VM_NAME:-motlie-v1-25-base-iter}"
SOURCE_IMAGE="${MOTLIE_VZ_SOURCE_IMAGE:-ghcr.io/cirruslabs/ubuntu@sha256:1e23e6fe5a6d3fb2089652229a09d71742617758b15aa311cecf1c05985d3021}"
RUN_LOG="$ARTIFACTS_DIR/build-run.log"
RESULT_JSON="$ARTIFACTS_DIR/build-result.json"
SOURCE_TARBALL="$ARTIFACTS_DIR/motlie-src.tar.gz"
TIMEOUT_SECONDS="${MOTLIE_VZ_TIMEOUT_SECONDS:-300}"
GUEST_SRC_DIR="/home/admin/motlie-src"
BOOTSTRAP_USER="${MOTLIE_VZ_BOOTSTRAP_USER:-admin}"
BOOTSTRAP_PASS="${MOTLIE_VZ_BOOTSTRAP_PASS:-admin}"
SERVICE_FILE="$SCRIPT_DIR/motlie-vfs-guest.service"
DATASOURCE_CFG_FILE="$SCRIPT_DIR/99_motlie_vz.cfg"

zmodload zsh/datetime

mkdir -p "$ARTIFACTS_DIR"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

require_cmd tart
require_cmd git
require_cmd tar
require_cmd python3
require_cmd expect
require_cmd scp
require_cmd ssh
require_cmd nc

local_vm_exists() {
  tart list --source local -q 2>/dev/null | grep -Fx "$1" >/dev/null 2>&1
}

cleanup() {
  if local_vm_exists "$BASE_VM_NAME"; then
    tart stop "$BASE_VM_NAME" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

echo "=== Vz v1.25 base guest build ==="
echo "Base VM:      $BASE_VM_NAME"
echo "Source image: $SOURCE_IMAGE"

if local_vm_exists "$BASE_VM_NAME"; then
  echo "--- deleting stale base VM clone ---"
  tart delete "$BASE_VM_NAME" >/dev/null || true
fi

echo "--- cloning base image ---"
tart clone "$SOURCE_IMAGE" "$BASE_VM_NAME" >/dev/null

echo "--- packing Motlie source tree ---"
COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 git -C "$REPO_ROOT" ls-files -z | COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 tar --disable-copyfile --no-mac-metadata --no-xattrs --null -czf "$SOURCE_TARBALL" -C "$REPO_ROOT" --files-from -

echo "--- starting guest ---"
: > "$RUN_LOG"
START_EPOCH="$EPOCHREALTIME"
tart run --no-graphics "$BASE_VM_NAME" >"$RUN_LOG" 2>&1 &
RUN_PID="$!"

IP_ADDR=""
ATTEMPTS=0
MAX_ATTEMPTS=$(( TIMEOUT_SECONDS * 2 ))
while [[ $ATTEMPTS -lt $MAX_ATTEMPTS ]]; do
  if ! kill -0 "$RUN_PID" >/dev/null 2>&1; then
    echo "tart run exited early; log follows:" >&2
    cat "$RUN_LOG" >&2
    exit 1
  fi

  IP_ADDR="$(tart ip "$BASE_VM_NAME" 2>/dev/null || true)"
  if [[ -n "$IP_ADDR" ]]; then
    break
  fi

  sleep 0.5
  ATTEMPTS=$(( ATTEMPTS + 1 ))
done

if [[ -z "$IP_ADDR" ]]; then
  echo "timed out waiting for guest IP after ${TIMEOUT_SECONDS}s" >&2
  cat "$RUN_LOG" >&2 || true
  exit 1
fi

echo "--- waiting for guest SSH ---"
ATTEMPTS=0
while [[ $ATTEMPTS -lt $MAX_ATTEMPTS ]]; do
  if nc -z "$IP_ADDR" 22 >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
  ATTEMPTS=$(( ATTEMPTS + 1 ))
done
if [[ $ATTEMPTS -ge $MAX_ATTEMPTS ]]; then
  echo "timed out waiting for guest SSH after ${TIMEOUT_SECONDS}s" >&2
  exit 1
fi

READY_EPOCH="$EPOCHREALTIME"
BOOT_SECONDS="$(awk -v start="$START_EPOCH" -v ready="$READY_EPOCH" 'BEGIN { printf "%.3f", ready - start }')"

guest_bash() {
  local remote_script
  remote_script="$(mktemp)"
  cat >"$remote_script"
  expect <<EOF
set timeout -1
set password_tries 0
spawn scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$remote_script" ${BOOTSTRAP_USER}@${IP_ADDR}:/tmp/motlie-vnet-remote.sh
expect {
  "password:" {
    incr password_tries
    if {\$password_tries > 3} {
      puts stderr "scp auth failed for ${BOOTSTRAP_USER}@${IP_ADDR}"
      exit 97
    }
    send "${BOOTSTRAP_PASS}\r"
    exp_continue
  }
  "Permission denied" {
    puts stderr "scp permission denied for ${BOOTSTRAP_USER}@${IP_ADDR}"
    exit 98
  }
  eof {
    catch wait result
    set exit_code [lindex \$result 3]
    if {\$exit_code != 0} {
      exit \$exit_code
    }
  }
}
EOF
  rm -f "$remote_script"
  expect <<EOF
set timeout -1
set password_tries 0
  spawn ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${BOOTSTRAP_USER}@${IP_ADDR} "bash -euo pipefail /tmp/motlie-vnet-remote.sh </dev/null"
expect {
  "password:" {
    incr password_tries
    if {\$password_tries > 3} {
      puts stderr "ssh auth failed for ${BOOTSTRAP_USER}@${IP_ADDR}"
      exit 97
    }
    send "${BOOTSTRAP_PASS}\r"
    exp_continue
  }
  "Permission denied" {
    puts stderr "ssh permission denied for ${BOOTSTRAP_USER}@${IP_ADDR}"
    exit 98
  }
  eof {
    catch wait result
    set exit_code [lindex \$result 3]
    if {\$exit_code != 0} {
      exit \$exit_code
    }
  }
}
EOF
}

guest_copy() {
  local src="$1"
  local dst="$2"
  expect <<EOF
set timeout -1
set password_tries 0
spawn scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$src" ${BOOTSTRAP_USER}@${IP_ADDR}:$dst
expect {
  "password:" {
    incr password_tries
    if {\$password_tries > 3} {
      puts stderr "scp auth failed for ${BOOTSTRAP_USER}@${IP_ADDR}"
      exit 97
    }
    send "${BOOTSTRAP_PASS}\r"
    exp_continue
  }
  "Permission denied" {
    puts stderr "scp permission denied for ${BOOTSTRAP_USER}@${IP_ADDR}"
    exit 98
  }
  eof {
    catch wait result
    set exit_code [lindex \$result 3]
    if {\$exit_code != 0} {
      exit \$exit_code
    }
  }
}
EOF
}

guest_fetch() {
  local src="$1"
  local dst="$2"
  expect <<EOF
set timeout -1
set password_tries 0
spawn scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${BOOTSTRAP_USER}@${IP_ADDR}:$src "$dst"
expect {
  "password:" {
    incr password_tries
    if {\$password_tries > 3} {
      puts stderr "scp auth failed for ${BOOTSTRAP_USER}@${IP_ADDR}"
      exit 97
    }
    send "${BOOTSTRAP_PASS}\r"
    exp_continue
  }
  "Permission denied" {
    puts stderr "scp permission denied for ${BOOTSTRAP_USER}@${IP_ADDR}"
    exit 98
  }
  eof {
    catch wait result
    set exit_code [lindex \$result 3]
    if {\$exit_code != 0} {
      exit \$exit_code
    }
  }
}
EOF
}

guest_capture() {
  local remote_script
  remote_script="$(mktemp)"
  cat >"$remote_script"
  expect <<EOF
set timeout -1
spawn scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$remote_script" ${BOOTSTRAP_USER}@${IP_ADDR}:/tmp/motlie-vnet-capture.sh
expect "password:"
send "${BOOTSTRAP_PASS}\r"
expect eof
catch wait result
set exit_code [lindex \$result 3]
if {\$exit_code != 0} {
  exit \$exit_code
}
EOF
  rm -f "$remote_script"
  expect <<EOF
set timeout -1
log_user 0
set output ""
set password_tries 0
spawn ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${BOOTSTRAP_USER}@${IP_ADDR} "bash -euo pipefail /tmp/motlie-vnet-capture.sh </dev/null"
expect {
  "password:" {
    incr password_tries
    if {\$password_tries > 3} {
      puts stderr "ssh auth failed for ${BOOTSTRAP_USER}@${IP_ADDR}"
      exit 97
    }
    send "${BOOTSTRAP_PASS}\r"
    exp_continue
  }
  "Permission denied" {
    puts stderr "ssh permission denied for ${BOOTSTRAP_USER}@${IP_ADDR}"
    exit 98
  }
  -re ".+" {
    append output \$expect_out(0,string)
    exp_continue
  }
  eof {
    if {\$output ne ""} {
      puts \$output
    }
  }
}
catch wait result
set exit_code [lindex \$result 3]
if {\$exit_code != 0} {
  exit \$exit_code
}
EOF
}

tart_guest_bash() {
  local run_user="$1"
  local remote_script
  remote_script="$(mktemp)"
  cat >"$remote_script"
  guest_copy "$remote_script" /tmp/motlie-vnet-tart.sh
  rm -f "$remote_script"
  guest_bash <<'EOF'
sudo chmod 0644 /tmp/motlie-vnet-tart.sh
EOF
  if [[ "$run_user" == "admin" ]]; then
    tart exec "$BASE_VM_NAME" bash -euo pipefail /tmp/motlie-vnet-tart.sh
  else
    tart exec "$BASE_VM_NAME" sudo -u "$run_user" bash -euo pipefail /tmp/motlie-vnet-tart.sh
  fi
}

tart_guest_capture() {
  local run_user="$1"
  local remote_script
  remote_script="$(mktemp)"
  cat >"$remote_script"
  guest_copy "$remote_script" /tmp/motlie-vnet-tart-capture.sh
  rm -f "$remote_script"
  guest_bash <<'EOF'
sudo chmod 0644 /tmp/motlie-vnet-tart-capture.sh
EOF
  if [[ "$run_user" == "admin" ]]; then
    tart exec "$BASE_VM_NAME" bash -euo pipefail /tmp/motlie-vnet-tart-capture.sh
  else
    tart exec "$BASE_VM_NAME" sudo -u "$run_user" bash -euo pipefail /tmp/motlie-vnet-tart-capture.sh
  fi
}

echo "--- installing guest prerequisites ---"
guest_bash <<'EOF'
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  pkg-config \
  libfuse3-dev \
  libfuse3-3 \
  fuse3 \
  curl \
  ca-certificates \
  tar \
  gzip \
  iproute2 \
  tmux \
  locales \
  bubblewrap \
  dnsutils \
  python3 \
  npm
sudo sed -i '/^en_US.UTF-8 UTF-8$/d' /etc/locale.gen
printf 'en_US.UTF-8 UTF-8\n' | sudo tee -a /etc/locale.gen >/dev/null
sudo locale-gen en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8
EOF

echo "--- installing Rust toolchain in guest if needed ---"
guest_bash <<'EOF'
if ! command -v cargo >/dev/null 2>&1; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y
fi
EOF

echo "--- uploading Motlie source tree into guest ---"
guest_copy "$SOURCE_TARBALL" /tmp/motlie-src.tar.gz
guest_copy "$SERVICE_FILE" /tmp/motlie-vfs-guest.service
guest_copy "$DATASOURCE_CFG_FILE" /tmp/99_motlie_vz.cfg
guest_bash <<EOF
rm -rf '$GUEST_SRC_DIR'
mkdir -p '$GUEST_SRC_DIR'
tar -xzf /tmp/motlie-src.tar.gz -C '$GUEST_SRC_DIR'
EOF

echo "--- building guest binaries and CLIs in guest ---"
guest_bash <<EOF
export PATH="\$HOME/.cargo/bin:\$PATH"
export CARGO_TARGET_DIR="\$HOME/motlie-target"
python3 - <<'PY'
from pathlib import Path

root = Path("$GUEST_SRC_DIR/Cargo.toml")
text = root.read_text(encoding="utf-8")
if '"libs/vfs",' not in text:
    text = text.replace('members = [\n', 'members = [\n    "libs/vfs",\n', 1)
    root.write_text(text, encoding="utf-8")
PY
cargo build --manifest-path '$GUEST_SRC_DIR/libs/vfs/Cargo.toml' --release --features vsock,client --bin motlie-vfs-guest-v1_1
sudo npm install -g @openai/codex
sudo npm install -g @anthropic-ai/claude-code
EOF

echo "--- installing converged v1.25 guest contract ---"
guest_bash <<'EOF'
sudo install -D -m 0755 "$HOME/motlie-target/release/motlie-vfs-guest-v1_1" /usr/local/bin/motlie-vfs-guest
sudo install -D -m 0644 /tmp/motlie-vfs-guest.service /etc/systemd/system/motlie-vfs-guest.service
sudo install -D -m 0644 /tmp/99_motlie_vz.cfg /etc/cloud/cloud.cfg.d/99_motlie_vz.cfg
sudo mkdir -p /etc/motlie-vfs
sudo mkdir -p /etc/profile.d
EOF

echo "--- creating bootstrap user for identity remap ---"
tart_guest_bash admin <<'EOF'
if ! id -u motlie-build >/dev/null 2>&1; then
    sudo groupadd -f -g 2002 motlie-build
    sudo useradd -m -u 2002 -g 2002 -s /bin/bash motlie-build
fi
echo "motlie-build:admin" | sudo chpasswd
sudo usermod -aG sudo motlie-build || true
printf '%s\n' 'motlie-build ALL=(ALL) NOPASSWD:ALL' | sudo tee /etc/sudoers.d/90-motlie-build >/dev/null
sudo chown root:root /etc/sudoers.d/90-motlie-build
sudo chmod 0440 /etc/sudoers.d/90-motlie-build
EOF

BOOTSTRAP_READY="$(tart_guest_capture admin <<'EOF'
if id -u motlie-build >/dev/null 2>&1 && sudo -u motlie-build sudo -n true >/dev/null 2>&1; then
  echo ok
fi
EOF
)"
if [[ "${BOOTSTRAP_READY##*$'\n'}" != "ok" && "$BOOTSTRAP_READY" != "ok" ]]; then
  echo "motlie-build bootstrap user is not ready for passwordless sudo" >&2
  exit 1
fi

tart_guest_bash motlie-build <<'EOF'

remap_conflicting_identity() {
    user_name="$1"
    target_uid="$2"
    target_gid="$3"
    remap_uid="$4"
    remap_gid="$5"

    current_uid="$(id -u "$user_name" 2>/dev/null || true)"
    current_gid="$(getent group "$user_name" | cut -d: -f3 || true)"

    if [ "$current_uid" = "$target_uid" ]; then
        sudo usermod -u "$remap_uid" "$user_name"
        sudo find / -xdev -uid "$target_uid" -exec chown -h "$remap_uid" {} + 2>/dev/null || true
    fi
    if [ "$current_gid" = "$target_gid" ]; then
        sudo groupmod -g "$remap_gid" "$user_name"
        sudo find / -xdev -gid "$target_gid" -exec chgrp -h "$remap_gid" {} + 2>/dev/null || true
    fi
}

ensure_guest_identity() {
    user_name="$1"
    target_uid="$2"
    target_gid="$3"
    password="$4"

    existing_gid="$(getent group "$user_name" | cut -d: -f3 || true)"
    if [ -z "$existing_gid" ]; then
        gid_owner="$(getent group "$target_gid" | cut -d: -f1 || true)"
        if [ -n "$gid_owner" ] && [ "$gid_owner" != "$user_name" ]; then
            echo "gid $target_gid already belongs to $gid_owner" >&2
            exit 1
        fi
        sudo groupadd -g "$target_gid" "$user_name"
    elif [ "$existing_gid" != "$target_gid" ]; then
        echo "group $user_name has gid $existing_gid but expected $target_gid" >&2
        exit 1
    fi

    existing_uid="$(id -u "$user_name" 2>/dev/null || true)"
    if [ -z "$existing_uid" ]; then
        uid_owner="$(getent passwd "$target_uid" | cut -d: -f1 || true)"
        if [ -n "$uid_owner" ] && [ "$uid_owner" != "$user_name" ]; then
            echo "uid $target_uid already belongs to $uid_owner" >&2
            exit 1
        fi
        sudo useradd -m -u "$target_uid" -g "$target_gid" -s /bin/bash "$user_name"
    elif [ "$existing_uid" != "$target_uid" ]; then
        echo "user $user_name has uid $existing_uid but expected $target_uid" >&2
        exit 1
    fi

    sudo usermod -aG sudo "$user_name" || true
    echo "$user_name:$password" | sudo chpasswd
}

remap_conflicting_identity admin 1000 1000 2000 2000
remap_conflicting_identity ubuntu 1001 1001 2001 2001
ensure_guest_identity alice 1000 1000 testpass
ensure_guest_identity bob 1001 1001 testpass

cat <<'TMUXEOF' | sudo tee /etc/profile.d/tmux-auto.sh >/dev/null
if [ -n "$SSH_CONNECTION" ] && [ -z "$TMUX" ] && command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t "$USER" 2>/dev/null; then
        echo "Attaching to existing tmux session..."
        sleep 1
        exec tmux attach-session -t "$USER"
    else
        printf "Start tmux session? [Y/n] (auto-yes in 3s) "
        if read -r -n 1 -t 3 answer; then
            echo
        else
            answer=Y
            echo
        fi
        case "$answer" in
            n|N) ;;
            *) exec tmux new-session -s "$USER" ;;
        esac
    fi
fi
TMUXEOF
cat <<'DOTENVEOF' | sudo tee /etc/profile.d/dotenv.sh >/dev/null
if [ -f "$HOME/.env" ]; then
    set -a
    . "$HOME/.env"
    set +a
fi
DOTENVEOF
cat <<'AGENTEOF' | sudo tee /etc/profile.d/agent-state.sh >/dev/null
agent_state_root=/agent-state
codex_root="$agent_state_root/codex"
codex_sqlite_root="$codex_root/sqlite"
claude_root="$agent_state_root/claude"
claude_code_root="$agent_state_root/claude-code"
if [ -d "$agent_state_root" ] && [ -n "${HOME:-}" ] && [ -d "$HOME" ] && [ "${USER:-}" != "root" ] && [ "${HOME#"/home/"}" != "$HOME" ]; then
    mkdir -p "$codex_root" "$codex_sqlite_root" "$claude_root" "$claude_code_root" "$HOME/.config" >/dev/null 2>&1 || true
    export CODEX_HOME="$codex_root"
    export CODEX_SQLITE_HOME="$codex_sqlite_root"
fi
AGENTEOF
cat <<'AGENTSVCEOF' | sudo tee /usr/local/bin/motlie-agent-state-setup >/dev/null
#!/bin/sh
set -eu

setup_user() {
    user_name="$1"
    home_dir="/home/$user_name"

    [ -d "$home_dir" ] || return 0

    install -d -m 0755 "$home_dir/.config"
    install -d -m 0700 /agent-state/codex /agent-state/claude /agent-state/claude-code /agent-state/codex/sqlite

    chown -R "$user_name:$user_name" \
        "$home_dir/.config" \
        /agent-state/codex \
        /agent-state/codex/sqlite \
        /agent-state/claude \
        /agent-state/claude-code || true

    rm -rf "$home_dir/.codex" "$home_dir/.claude" "$home_dir/.config/claude-code"
    ln -sfn /agent-state/codex "$home_dir/.codex"
    ln -sfn /agent-state/claude "$home_dir/.claude"
    ln -sfn /agent-state/claude-code "$home_dir/.config/claude-code"
    chown -h "$user_name:$user_name" \
        "$home_dir/.codex" \
        "$home_dir/.claude" \
        "$home_dir/.config/claude-code" || true
}

for user_name in alice bob; do
    if id -u "$user_name" >/dev/null 2>&1; then
        setup_user "$user_name"
    fi
done
AGENTSVCEOF
cat <<'MOTDEOF' | sudo tee /etc/motd >/dev/null
                    _   _ _
  _ __ ___   ___ | |_| (_) ___
 | '_ ` _ \ / _ \| __| | |/ _ \
 | | | | | | (_) | |_| | |  __/
 |_| |_| |_|\___/ \__|_|_|\___|

v1.25 Apple Vz vnet / agent-state demo
MOTDEOF
if ! grep -qx 'user_allow_other' /etc/fuse.conf 2>/dev/null; then
  printf 'user_allow_other\n' | sudo tee -a /etc/fuse.conf >/dev/null
fi
sudo chmod 0644 /etc/profile.d/tmux-auto.sh /etc/profile.d/dotenv.sh /etc/profile.d/agent-state.sh
sudo chmod 0755 /usr/local/bin/motlie-agent-state-setup
cat <<'AGENTUNITEOF' | sudo tee /etc/systemd/system/motlie-agent-state.service >/dev/null
[Unit]
Description=Link agent state into mounted guest home
After=motlie-vfs-guest.service
Requires=motlie-vfs-guest.service
ConditionPathIsDirectory=/agent-state

[Service]
Type=oneshot
ExecStart=/usr/local/bin/motlie-agent-state-setup
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
AGENTUNITEOF
sudo systemctl unmask motlie-vfs-guest.service || true
sudo systemctl daemon-reload
sudo systemctl enable motlie-vfs-guest.service >/dev/null 2>&1 || true
sudo systemctl enable motlie-agent-state.service >/dev/null 2>&1 || true
EOF

echo "--- cleaning cloud-init state for reusable base image ---"
tart_guest_bash admin <<'EOF'
sudo cloud-init clean --logs --machine-id --seed
sudo rm -rf /var/lib/cloud/*
sudo truncate -s 0 /etc/machine-id
sudo mkdir -p /var/lib/cloud
EOF

GUEST_BINARY="/usr/local/bin/motlie-vfs-guest"

IDENTITY_PAYLOAD="$(tart_guest_capture admin <<'EOF'
python3 - <<'PY'
import json
import pwd
import grp

def passwd_entry(name: str):
    try:
        entry = pwd.getpwnam(name)
        return {"name": entry.pw_name, "uid": entry.pw_uid, "gid": entry.pw_gid, "home": entry.pw_dir}
    except KeyError:
        return None

def group_entry(name: str):
    try:
        entry = grp.getgrnam(name)
        return {"name": entry.gr_name, "gid": entry.gr_gid}
    except KeyError:
        return None

payload = {
    "passwd": {
        "admin": passwd_entry("admin"),
        "alice": passwd_entry("alice"),
        "bob": passwd_entry("bob"),
    },
    "group": {
        "admin": group_entry("admin"),
        "alice": group_entry("alice"),
        "bob": group_entry("bob"),
    },
}
print(json.dumps(payload, sort_keys=True))
PY
EOF
)"

python3 - "$RESULT_JSON" "$BASE_VM_NAME" "$IP_ADDR" "$BOOT_SECONDS" "$GUEST_BINARY" "$IDENTITY_PAYLOAD" <<'PY'
import json
import sys

path, vm_name, ip_addr, boot_seconds, guest_binary, identity_payload = sys.argv[1:]
with open(path, "w", encoding="utf-8") as fh:
    json.dump(
        {
            "backend": "vz-tart",
            "vm_name": vm_name,
            "ip_addr": ip_addr,
            "boot_to_ip_seconds": float(boot_seconds),
            "guest_contract": {
                "motlie_vfs_guest_path": guest_binary,
                "users": {
                    "alice": {"uid": 1000, "gid": 1000, "password": "testpass"},
                    "bob": {"uid": 1001, "gid": 1001, "password": "testpass"},
                },
                "agent_state": "/agent-state",
            },
            "identity_probe": json.loads(identity_payload),
        },
        fh,
        indent=2,
        sort_keys=True,
    )
    fh.write("\n")
PY

echo "--- result written ---"
cat "$RESULT_JSON"

echo "--- shutting down base guest gracefully ---"
tart exec "$BASE_VM_NAME" sudo shutdown -h now >/dev/null 2>&1 || true
wait "$RUN_PID" || true

echo "=== success ==="
