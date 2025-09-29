#!/usr/bin/env bash
# ------------------------------------------------------
# Log in to Azure interactively
# ------------------------------------------------------

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

set -a
source "$SCRIPT_DIR/../.env"
set +a

az login --tenant "$TENANT_ID"
