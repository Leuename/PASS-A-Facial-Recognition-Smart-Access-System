#!/bin/bash
set -e

cd "$(dirname "$0")"

if [ -d "arclight_env" ]; then
    source arclight_env/bin/activate
fi

PORT="${ARCLIGHT_PORT:-8000}"
LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"

if [ -z "$LAN_IP" ]; then
    LAN_IP="127.0.0.1"
fi

echo ""
echo "Arclight is starting."
echo "Open this URL in your browser:"
echo "http://${LAN_IP}:${PORT}"
echo ""

uvicorn arclight_server:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --log-level info
