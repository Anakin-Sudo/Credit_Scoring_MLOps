#!/usr/bin/env bash
# ------------------------------------------------------
# Assign role to Service Principal on AML workspace
# ------------------------------------------------------

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

set -a
source "$SCRIPT_DIR/../.env"
set +a

WORKSPACE_ID=$(az ml workspace show --name "$WORKSPACE_NAME" --query "id" -o tsv)

az role assignment create \
  --assignee "$SP_APP_ID" \
  --role "AzureML Data Scientist" \
  --scope "$WORKSPACE_ID"
