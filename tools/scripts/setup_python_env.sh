#!/bin/bash
# Sets up a Python virtual environment with required dependencies and
# configures PYTHONPATH so Bazel can find them.
#
# Usage:
#   source tools/scripts/setup_python_env.sh
#
# Or to specify a custom venv location:
#   VENV_DIR=/path/to/venv source tools/scripts/setup_python_env.sh

set -e

VENV_DIR="${VENV_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)/.venv}"
REQUIREMENTS="${REQUIREMENTS:-$(dirname "$BASH_SOURCE")/../../bindings/pyc3/requirements.txt}"

if [ ! -d "$VENV_DIR" ]; then
    echo "[setup_python_env] Creating virtualenv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR" --system-site-packages
fi

source "$VENV_DIR/bin/activate"

echo "[setup_python_env] Installing Python dependencies..."
pip install --quiet -r "$REQUIREMENTS"

# Export PYTHONPATH for Bazel's --action_env=PYTHONPATH
SITE_PACKAGES=$(python3 -c "import site; print(':'.join(site.getsitepackages()))")
USER_SITE=$(python3 -c "import site; print(site.getusersitepackages())")
export PYTHONPATH="${SITE_PACKAGES}:${USER_SITE}:${PYTHONPATH:-}"
export VIRTUAL_ENV="$VENV_DIR"

echo "[setup_python_env] Done. PYTHONPATH=$PYTHONPATH"
