from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

vault_url = "https://<your-key-vault-name>.vault.azure.net/"
client = SecretClient(vault_url=vault_url, credential=DefaultAzureCredential())

conn_str = client.get_secret("storage-conn-string").value


