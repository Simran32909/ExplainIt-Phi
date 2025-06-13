import os
from datasets import load_dataset
import json

def parse_and_split_st_eli5(output_dir="data/processed", test_size=0.1, val_size=0.1):
    """
    Loads the sentence-transformers/eli5 dataset, parses the text field into
    instruction-output pairs, splits it into train, validation, and test sets,
    and saves them as JSON files.
    """
    print("Loading sentence-transformers/eli5 dataset...")
    # Load the training split of the dataset
    dataset = load_dataset("sentence-transformers/eli5", split="train")

    def parse_text_to_instruction_output(example):
        text = example['text']
        # The text is structured as "question: [Question] answer: [Answer]"
        # We'll split based on " answer: "
        parts = text.split(" answer: ")
        if len(parts) == 2:
            question_part, answer_part = parts
            # Remove the "question: " prefix from the question part
            if question_part.startswith("question: "):
                question = question_part[len("question: "):].strip()
                instruction = f"Explain the following like I'm 5: {question}"
                output = answer_part.strip()
                return {"instruction": instruction, "output": output}
        
        # Return empty fields if parsing fails, to be filtered out later
        return {"instruction": "", "output": ""}

    print("Parsing and formatting dataset...")
    formatted_dataset = dataset.map(parse_text_to_instruction_output, batched=False)
    
    # Filter out any examples that failed parsing
    original_size = len(formatted_dataset)
    formatted_dataset = formatted_dataset.filter(lambda x: x['instruction'] and x['output'])
    new_size = len(formatted_dataset)
    print(f"Filtered out {original_size - new_size} examples that couldn't be parsed.")

    # Remove original columns
    formatted_dataset = formatted_dataset.remove_columns(['text'])
    
    # Shuffle the dataset before splitting
    shuffled_dataset = formatted_dataset.shuffle(seed=42)

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
    parse_and_split_st_eli5() 