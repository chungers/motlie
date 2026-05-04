# Shared v1.5 convergence constants for both CH Bash and VZ Zsh scripts.
#
# Keep this file shell-simple and sourceable by both Bash and Zsh. It marks the
# future Rust extraction boundary: constants and small render/verify helpers
# here become typed image-builder/launcher structs when the scripts converge.

MOTLIE_V15_CONTRACT_VERSION="${MOTLIE_V15_CONTRACT_VERSION:-v1.5}"
MOTLIE_V15_GUEST_MOUNTER_MARKER="${MOTLIE_V15_GUEST_MOUNTER_MARKER:-MOTLIE_VMM_GUEST_MOUNTER_V1_5}"
MOTLIE_V15_GUEST_BUILD_FEATURES="${MOTLIE_V15_GUEST_BUILD_FEATURES:---no-default-features --features guest-vfs}"
MOTLIE_V15_GUEST_BIN_OPT="${MOTLIE_V15_GUEST_BIN_OPT:-/opt/motlie/v1.5/guest/bin/motlie-vfs-guest}"
MOTLIE_V15_GUEST_BIN_COMPAT="${MOTLIE_V15_GUEST_BIN_COMPAT:-/usr/local/bin/motlie-vfs-guest}"
MOTLIE_V15_BACKEND_ENV_PATH="${MOTLIE_V15_BACKEND_ENV_PATH:-/etc/motlie/v1.5/backend.env}"
MOTLIE_V15_MOUNTS_PATH="${MOTLIE_V15_MOUNTS_PATH:-/etc/motlie-vfs/mounts.yaml}"
MOTLIE_V15_VFS_HOST_CID="${MOTLIE_V15_VFS_HOST_CID:-2}"
MOTLIE_V15_VFS_PORT="${MOTLIE_V15_VFS_PORT:-5000}"
MOTLIE_V15_SSH_VSOCK_PORT="${MOTLIE_V15_SSH_VSOCK_PORT:-2222}"

motlie_v15_ch_net_backend() {
  local egress_net="${1:-}"
  case "$egress_net" in
    vhost-user) printf '%s\n' "ch-vhost-user" ;;
    tap) printf '%s\n' "ch-tap" ;;
    none|"") printf '%s\n' "none" ;;
    *) printf '%s\n' "ch-${egress_net}" ;;
  esac
}

motlie_v15_write_backend_env() {
  local output_path="$1"
  local backend="$2"
  local net_backend="$3"

  mkdir -p "$(dirname "$output_path")"
  cat > "$output_path" <<EOF
# Rendered by examples/v1.5 common-contract.sh.
# Common guest-visible schema; backend-specific values are the adaptation.
MOTLIE_BACKEND=${backend}
MOTLIE_VFS_TRANSPORT=vsock
MOTLIE_VFS_HOST_CID=${MOTLIE_V15_VFS_HOST_CID}
MOTLIE_VFS_PORT=${MOTLIE_V15_VFS_PORT}
MOTLIE_VFS_CONNECT_TIMEOUT_MS=60000
MOTLIE_VFS_CONNECT_RETRY_MS=250
MOTLIE_NET_BACKEND=${net_backend}
MOTLIE_SSH_VSOCK_PORT=${MOTLIE_V15_SSH_VSOCK_PORT}
EOF
}

motlie_v15_require_guest_contract_json() {
  local contract_json="$1"
  local artifact_label="${2:-guest artifacts}"

  if [ ! -s "$contract_json" ]; then
    cat >&2 <<EOF
ERROR: ${artifact_label} missing v1.5 guest contract metadata:
  ${contract_json}

Do not reuse v1.4/v1.45 guest artifacts in examples/v1.5. Rebuild through the
v1.5 common image builder so the guest mounter is VMM-owned and built once.
EOF
    return 1
  fi

  if ! grep -q "$MOTLIE_V15_GUEST_MOUNTER_MARKER" "$contract_json"; then
    echo "ERROR: ${contract_json} does not record ${MOTLIE_V15_GUEST_MOUNTER_MARKER}" >&2
    return 1
  fi
  if ! grep -q -- "$MOTLIE_V15_GUEST_BUILD_FEATURES" "$contract_json"; then
    echo "ERROR: ${contract_json} does not record ${MOTLIE_V15_GUEST_BUILD_FEATURES}" >&2
    return 1
  fi
}
