#!/usr/bin/env bash
# ------------------------------------------------------
# Create an Azure ML compute cluster using .env values
# ------------------------------------------------------

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Load .env variables
set -a
source "$SCRIPT_DIR/../.env"
set +a

echo "Creating compute cluster $COMPUTE_NAME in workspace $WORKSPACE_NAME..."

az ml compute create \
  --name "$COMPUTE_NAME" \
  --type amlcompute \
  --size "$VM_SIZE" \
  --min-instances "$MIN_INSTANCES" \
  --max-instances "$MAX_INSTANCES" \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE_NAME"

echo "Compute cluster $COMPUTE_NAME created successfully."
