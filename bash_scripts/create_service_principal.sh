#!/usr/bin/env bash
# ------------------------------------------------------
# Create a Service Principal for CI/CD
# ------------------------------------------------------

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

set -a
source "$SCRIPT_DIR/../.env"
set +a

az ad sp create-for-rbac \
  --name "$SP_NAME" \
  --role Contributor \
  --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE_NAME \
  --sdk-auth > sp_output.json
