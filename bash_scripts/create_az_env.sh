#!/bin/bash

# Create or update an Azure ML environment
# The definition is taken from environments/env.yml
# Make sure this script is run from the project root
# (or adjust the path accordingly)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPT_DIR_WIN=$(wslpath -m "$SCRIPT_DIR")

az ml environment create --file "$SCRIPT_DIR_WIN/../environments/env.yml"

