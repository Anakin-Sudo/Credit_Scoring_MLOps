#!/usr/bin/env bash
# ------------------------------------------------------
# Create Azure ML workspace (requires az defaults set)
# ------------------------------------------------------

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

set -a
source "$SCRIPT_DIR/../.env"
set +a

az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

az ml workspace create --name "$WORKSPACE_NAME" --location "$LOCATION"
