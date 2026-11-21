#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
"${PWD}/.venv/bin/python" -m streamlit run lab/dashboard.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
