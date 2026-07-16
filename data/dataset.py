import json
import random
 
from datasets import Dataset, DatasetDict
 
from data.preprocessing import build_input_text
 
 
def load_relations(dataset_path):
    """Load the raw list of relation examples from the dataset json file."""
    with open(dataset_path) as f:
        return json.load(f)
 
 
def build_examples(relations, task_prefix):
    """Turn raw relation dicts into (inputs, entities, targets) lists."""
    inputs = []
    entities = []
    targets = []
 
    for rel in relations:
        input_text = build_input_text(rel["sentence"], rel["entities"], task_prefix)
        inputs.append(input_text)
        entities.append(rel["entities"])
        targets.append(rel["relation_text"])
 
    return inputs, entities, targets
 
 
def split_examples(inputs, entities, targets, seed, train_frac, val_frac):
    """Shuffle and split (inputs, entities, targets) into train/val/test.
 
    `train_frac` and `val_frac` are cumulative fractions of the full dataset
    (e.g. 0.8 / 0.9 gives an 80/10/10 train/val/test split).
    """
    n = len(inputs)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
 
    inputs = [inputs[i] for i in indices]
    entities = [entities[i] for i in indices]
    targets = [targets[i] for i in indices]
 
    train_split = int(train_frac * n)
    val_split = int(val_frac * n)
 
    splits = {}
    for name, lo, hi in [
        ("train", 0, train_split),
        ("validation", train_split, val_split),
        ("test", val_split, n),
    ]:
        splits[name] = {
            "inputs": inputs[lo:hi],
            "targets": targets[lo:hi],
        }
 
    return splits
 
 
def make_hf_dataset(inputs, targets):
    """Wrap parallel input/target lists into a single HF `Dataset`."""
    return Dataset.from_dict({"input_text": inputs, "target_text": targets})
 
 
def make_dataset_dict(splits):
    """Build a `DatasetDict` with train/validation/test splits from `split_examples` output."""
    return DatasetDict(
        {
            name: make_hf_dataset(split["inputs"], split["targets"])
            for name, split in splits.items()
        }
    )
 
 
def tokenize_dataset(dataset_dict, tokenizer, max_length):
    """Tokenize a raw (input_text, target_text) `DatasetDict` for seq2seq training.
 
    Pads labels with -100 in place of the tokenizer's pad token, so the loss
    ignores padding positions.
    """
 
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=max_length,
            truncation=True,
        )
        labels = tokenizer(
            examples["target_text"],
            max_length=max_length,
            truncation=True,
        )
        labels["input_ids"] = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
 
    return dataset_dict.map(
        preprocess_function,
        batched=True,
        remove_columns=["input_text", "target_text"],
    )
 
 
def load_and_prepare_datasets(dataset_path, tokenizer, task_prefix, max_length, seed, train_frac, val_frac):
    """Run the full pipeline: load raw json -> build inputs -> split -> tokenize.
 
    Returns the tokenized `DatasetDict` ready to hand to `Seq2SeqTrainer`.
    """
    relations = load_relations(dataset_path)
    inputs, entities, targets = build_examples(relations, task_prefix)
    splits = split_examples(inputs, entities, targets, seed, train_frac, val_frac)
    dataset_dict = make_dataset_dict(splits)
    return tokenize_dataset(dataset_dict, tokenizer, max_length)