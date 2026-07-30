import json
from datasets import Dataset, DatasetDict

from data.t5_preprocessing import build_input_text


def load_split(path):
    """Load one split file (a list of raw relation dicts) written by
    `data.splitting.prepare_data_splits`."""
    with open(path) as f:
        return json.load(f)


def build_examples(relations, task_prefix):
    """Turn raw relation dicts into (inputs, targets) lists."""
    inputs = []
    targets = []

    for rel in relations:
        input_text = build_input_text(rel["sentence"], rel["entities"], task_prefix)
        inputs.append(input_text)
        targets.append(rel["relation_text"])

    return inputs, targets


def make_hf_dataset(inputs, targets):
    """Wrap parallel input/target lists into a single HF `Dataset`."""
    return Dataset.from_dict({"input_text": inputs, "target_text": targets})


def make_dataset_dict(split_paths, task_prefix):
    """Build a raw (input_text, target_text) `DatasetDict` from split files on disk."""
    dataset_dict = {}
    for name, path in split_paths.items():
        relations = load_split(path)
        inputs, targets = build_examples(relations, task_prefix)
        print(f"{name}: {len(relations)} relations -> {len(inputs)} examples")
        dataset_dict[name] = make_hf_dataset(inputs, targets)
        print(f"{name}: dataset_dict length = {len(dataset_dict[name])}")
    return DatasetDict(dataset_dict)


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


def load_and_prepare_datasets(split_paths, tokenizer, task_prefix, max_length):
    """Run the T5-specific pipeline: load split files -> mark entities/build
    input-target text -> tokenize.

    `split_paths` is the dict of split name -> file path returned by
    `data.splitting.prepare_data_splits` (or `write_splits`). Returns the
    tokenized `DatasetDict` ready to hand to `Seq2SeqTrainer`.
    """
    dataset_dict = make_dataset_dict(split_paths, task_prefix)
    return tokenize_dataset(dataset_dict, tokenizer, max_length)