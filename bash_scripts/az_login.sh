#!/usr/bin/env bash
# ------------------------------------------------------
# Log in to Azure with tenant context
# ------------------------------------------------------

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Load .env
set -a
source "$SCRIPT_DIR/../.env"
set +a

echo "Logging in to Azure..."
az login --tenant "$TENANT_ID"
