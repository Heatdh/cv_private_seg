#!/bin/bash

cd /www/backend

while true; do
    # Run server
    flask run --no-debugger --no-reload --port=808
    echo "Backend crashed with exit code $?.  Respawning.." >&2
    sleep 1
done