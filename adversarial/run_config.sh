#!/bin/bash

ENV_FILE="/workspaces/FLAdverSarialInput/adversarial/.env"

choosen_conf=$1
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi

# Create a temporary file to hold the modified content
TEMP_FILE=$(mktemp)

# Try updating COUNT in the temporary file and move it back to the original file
if grep -q "^COUNT=" "$ENV_FILE"; then
# Update COUNT in the temporary file
    sed "s/^COUNT=.*/COUNT="$choosen_conf"/" "$ENV_FILE" > "$TEMP_FILE" && mv "$TEMP_FILE" "$ENV_FILE"
    echo "COUNT updated to "$choosen_conf""
else
# If COUNT doesn't exist, append it to the .env file
    echo "COUNT="$choosen_conf"" >> "$ENV_FILE"
    echo "COUNT added with value $choosen_conf"
fi
flwr run --run-config "num_cpus"=8