#!/usr/bin/env bash
# ------------------------------------------------------
# Script to set Azure CLI defaults using .env file
# ------------------------------------------------------

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Load .env variables (exported automatically)
set -a
source "$SCRIPT_DIR/../.env"
set +a

# Configure Azure defaults
az configure --defaults \
  group="$RESOURCE_GROUP" \
  workspace="$WORKSPACE_NAME" \
  subscription="$SUBSCRIPTION_ID"

echo "Azure CLI defaults set:
   Resource group: $RESOURCE_GROUP
   Workspace:      $WORKSPACE_NAME
   Subscription:   $SUBSCRIPTION_ID"
