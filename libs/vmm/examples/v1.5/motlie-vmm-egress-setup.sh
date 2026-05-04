#!/bin/sh
set -eu

# MOTLIE_VMM_V15_CH_EGRESS_SETUP_V1
# CH-specific adaptation for the common v1.5 image contract. The immutable
# rootfs installs this once; launch-ch.sh only seeds /etc/motlie-vmm/egress.*
# values into the per-guest overlay. Do not move package installs, builds, or
# runtime repair into this boot path.

EGRESS_MAC_FILE="/etc/motlie-vmm/egress.mac"
EGRESS_IPV4_FILE="/etc/motlie-vmm/egress.ipv4"
EGRESS_GATEWAY_FILE="/etc/motlie-vmm/egress.gateway"
EGRESS_DNS_FILE="/etc/motlie-vmm/egress.dns"
EGRESS_IFACE=""

if [ ! -f "$EGRESS_MAC_FILE" ]; then
    echo "motlie-vmm-egress-setup: missing $EGRESS_MAC_FILE" >&2
    exit 1
fi

EGRESS_MAC="$(cat "$EGRESS_MAC_FILE")"
[ -n "$EGRESS_MAC" ] || {
    echo "motlie-vmm-egress-setup: empty egress MAC" >&2
    exit 1
}

EGRESS_IPV4="$(cat "$EGRESS_IPV4_FILE" 2>/dev/null || true)"
EGRESS_GATEWAY="$(cat "$EGRESS_GATEWAY_FILE" 2>/dev/null || true)"
EGRESS_DNS="$(cat "$EGRESS_DNS_FILE" 2>/dev/null || true)"
[ -n "$EGRESS_IPV4" ] || EGRESS_IPV4="10.0.2.15"
[ -n "$EGRESS_GATEWAY" ] || EGRESS_GATEWAY="10.0.2.2"
[ -n "$EGRESS_DNS" ] || EGRESS_DNS="10.0.2.3"
EGRESS_NETWORK="${EGRESS_IPV4%.*}.0"

for _attempt in $(seq 1 30); do
    for candidate in /sys/class/net/*; do
        [ -e "$candidate/address" ] || continue
        candidate_mac="$(cat "$candidate/address" 2>/dev/null || true)"
        if [ "$candidate_mac" = "$EGRESS_MAC" ]; then
            EGRESS_IFACE="$(basename "$candidate")"
            break
        fi
    done
    [ -n "$EGRESS_IFACE" ] && break
    sleep 1
done

[ -n "$EGRESS_IFACE" ] || {
    echo "motlie-vmm-egress-setup: interface with MAC $EGRESS_MAC not found" >&2
    exit 1
}

mkdir -p /run
ln -sf "/sys/class/net/$EGRESS_IFACE" /run/motlie-vmm-egress.link
ip link set "$EGRESS_IFACE" up || exit 0
ip addr replace "$EGRESS_IPV4/24" dev "$EGRESS_IFACE"
ip route replace "$EGRESS_NETWORK/24" dev "$EGRESS_IFACE" scope link src "$EGRESS_IPV4"
ip route replace default via "$EGRESS_GATEWAY" dev "$EGRESS_IFACE" metric 100
cat > /etc/resolv.conf <<EOF
nameserver $EGRESS_DNS
options edns0
EOF
