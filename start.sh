#!/usr/bin/env bash


if [[ $1 = "debug" ]]; then
   python3 -m debugpy --listen 5678 /app/main.py
elif [[ $1 = "profile" ]]; then
    kernprof -l /app/main.py
else
    # only use one worker because we have no message quque
    export MAX_WORKERS=1
    uvicorn main:node --host 0.0.0.0 --port 80 --reload --lifespan on
fi
