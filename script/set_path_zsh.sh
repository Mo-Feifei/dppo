#!/bin/zsh

##################### Paths #####################

# Set default paths
DEFAULT_DATA_DIR="${PWD}/data"
DEFAULT_LOG_DIR="${PWD}/log"

# Prompt the user for input, allowing overrides
read -p "Enter the desired data directory [default: ${DEFAULT_DATA_DIR}], leave empty to use default: " DATA_DIR
DPPO_DATA_DIR=${DATA_DIR:-$DEFAULT_DATA_DIR}  # Use user input or default if input is empty

read -p "Enter the desired logging directory [default: ${DEFAULT_LOG_DIR}], leave empty to use default: " LOG_DIR
DPPO_LOG_DIR=${LOG_DIR:-$DEFAULT_LOG_DIR}  # Use user input or default if input is empty

# Export to current session
export DPPO_DATA_DIR="$DPPO_DATA_DIR"
export DPPO_LOG_DIR="$DPPO_LOG_DIR"

# Confirm the paths with the user
echo "Data directory set to: $DPPO_DATA_DIR"
echo "Log directory set to: $DPPO_LOG_DIR"

# Append environment variables to .zshrc
echo "export DPPO_DATA_DIR=\"$DPPO_DATA_DIR\"" >> ~/.zshrc
echo "export DPPO_LOG_DIR=\"$DPPO_LOG_DIR\"" >> ~/.zshrc

echo "Environment variables DPPO_DATA_DIR and DPPO_LOG_DIR added to .zshrc and applied to the current session."

##################### WandB #####################

# Prompt the user for input, allowing overrides
read -p "Enter your WandB entity (username or team name), leave empty to skip: " ENTITY

# Check if ENTITY is not empty
if [ -n "$ENTITY" ]; then
  # If ENTITY is not empty, set the environment variable
  export DPPO_WANDB_ENTITY="$ENTITY"

  # Confirm the entity with the user
  echo "WandB entity set to: $DPPO_WANDB_ENTITY"

  # Append environment variable to .zshrc
  echo "export DPPO_WANDB_ENTITY=\"$ENTITY\"" >> ~/.zshrc
  
  echo "Environment variable DPPO_WANDB_ENTITY added to .zshrc and applied to the current session."
else
  # If ENTITY is empty, skip setting the environment variable
  echo "No WandB entity provided. Please set wandb=null when running scripts to disable wandb logging and avoid error."
fi
