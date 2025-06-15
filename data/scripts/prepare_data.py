import os
from datasets import load_dataset
import json

def prepare_and_split_eli5(output_dir="data/processed", test_size=0.1, val_size=0.1):
    """
    Loads the sentence-transformers/eli5 dataset, splits it into train,
    validation, and test sets, and saves them as JSON files.
    """
    print("Loading sentence-transformers/eli5 dataset...")
    # Load the training split of the dataset
    dataset = load_dataset("sentence-transformers/eli5", split="train[:10000]")

    # Shuffle the dataset before splitting
    shuffled_dataset = dataset.shuffle(seed=42)

    print("Splitting data into train, validation, and test sets...")
    # Create a 80% train / 20% test+validation split
    train_test_split = shuffled_dataset.train_test_split(test_size=(test_size + val_size))
    train_dataset = train_test_split['train']
    
    # Split the 20% into half for validation, half for testing (10% each of total)
    test_val_split = train_test_split['test'].train_test_split(test_size=0.5)
    validation_dataset = test_val_split['train']
    test_dataset = test_val_split['test']

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(validation_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the splits to separate JSON files
    def save_dataset_to_json(dataset, path):
        print(f"Saving dataset to {path}...")
        dataset_list = list(dataset)
        with open(path, "w", encoding='utf-8') as f:
            json.dump(dataset_list, f, indent=4)
        print(f"Saved {len(dataset_list)} records.")

    save_dataset_to_json(train_dataset, os.path.join(output_dir, "train.json"))
    save_dataset_to_json(validation_dataset, os.path.join(output_dir, "validation.json"))
    save_dataset_to_json(test_dataset, os.path.join(output_dir, "test.json"))
        
    print("Data preparation and splitting complete.")


if __name__ == "__main__":
    prepare_and_split_eli5() 