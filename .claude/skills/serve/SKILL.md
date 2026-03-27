---
name: serve
description: Start the Sentinel FastAPI server
user-invocable: true
argument-hint: "[port] (default: 8000)"
allowed-tools: Bash, Read
context: fork
---

# /serve $ARGUMENTS

Start the Sentinel FastAPI server.

## Instructions

1. Parse optional port argument (default: 8000)

2. Check if the API module exists:
   ```bash
   ls src/sentinel/api/app.py 2>/dev/null
   ```

3. If it exists, start the server:
   ```bash
   uv run sentinel serve --host 0.0.0.0 --port {port}
   ```

4. If it doesn't exist, report which phase implements it:
   ```
   API not implemented yet — this is Phase 8 (FastAPI + Dashboard).
   Current progress: /status
   ```

5. After starting, verify health:
   ```bash
   curl -s http://localhost:{port}/health 2>/dev/null || echo "Server starting..."
   ```
