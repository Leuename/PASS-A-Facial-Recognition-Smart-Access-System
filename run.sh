#!/bin/bash
set -e

cd "$(dirname "$0")"

if [ -d "arclight_env" ]; then
    source arclight_env/bin/activate
fi

uvicorn arclight_server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info
