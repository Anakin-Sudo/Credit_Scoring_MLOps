#!/usr/bin/env bash
# ------------------------------------------------------
# Assign Key Vault access to the compute's managed identity
# ------------------------------------------------------

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Load .env variables
set -a
source "$SCRIPT_DIR/../.env"
set +a

echo "Fetching principal ID for compute $COMPUTE_NAME..."

PRINCIPAL_ID=$(az ml compute show \
  --name "$COMPUTE_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE_NAME" \
  --query "identity.principal_id" -o tsv)

echo "Compute $COMPUTE_NAME has principal ID: $PRINCIPAL_ID"

echo "Fetching Key Vault resource ID..."
KV_ID=$(az keyvault show --name "$KEYVAULT_NAME" --query id -o tsv)

echo "Assigning 'Key Vault Secrets User' role on $KEYVAULT_NAME to compute $COMPUTE_NAME..."

az role assignment create \
  --assignee "$PRINCIPAL_ID" \
  --role "Key Vault Secrets User" \
  --scope "$KV_ID"

echo "Role assignment completed."
