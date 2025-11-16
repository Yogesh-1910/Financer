import json
import random
from pathlib import Path

# --- Configuration ---
INPUT_FILE = "finance_qa.json"
TRAIN_FILE = "finance_train.json"
VALIDATION_FILE = "finance_validation.json"
VALIDATION_SET_SIZE = 100  # Number of samples to hold out for validation

def split_dataset():
    """
    Reads a JSON dataset, shuffles it, and splits it into
    a training and a validation set to prevent data leaks.
    """
    # Ensure the input file exists
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        return

    print(f"Loading data from '{INPUT_FILE}'...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Shuffle the data to ensure the validation set is a random sample
    random.shuffle(data)

    # Check if the dataset is large enough for the split
    if len(data) <= VALIDATION_SET_SIZE:
        print(f"Warning: Dataset size ({len(data)}) is too small to create a validation set of size {VALIDATION_SET_SIZE}.")
        print("Using the entire dataset for training.")
        train_data = data
        validation_data = []
    else:
        # Split the data
        validation_data = data[:VALIDATION_SET_SIZE]
        train_data = data[VALIDATION_SET_SIZE:]
        print(f"Successfully split dataset:")
        print(f" - Training examples:   {len(train_data)}")
        print(f" - Validation examples: {len(validation_data)}")

    # Save the training set
    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)
    print(f"Training data saved to '{TRAIN_FILE}'")

    # Save the validation set
    if validation_data:
        with open(VALIDATION_FILE, 'w', encoding='utf-8') as f:
            json.dump(validation_data, f, indent=2)
        print(f"Validation data saved to '{VALIDATION_FILE}'")

if __name__ == "__main__":
    split_dataset()