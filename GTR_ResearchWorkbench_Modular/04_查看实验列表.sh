#!/usr/bin/env bash
cd "$(dirname "$0")"
"${PWD}/.venv/bin/python" -m lab.cli status
