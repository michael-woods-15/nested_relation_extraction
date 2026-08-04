import json
from datasets import Dataset, DatasetDict

from data.bert_data_creation import generate_training_examples


def load_split(path):
    """Load one split file (a list of raw nested-relation dicts) written by
    `data.splitting.prepare_data_splits`."""
    with open(path) as f:
        return json.load(f)


def build_bert_examples(relations):
    """Expand a list of raw nested-relation dicts into flat BERT
    `{"text": ..., "label": ...}` training examples."""
    examples = []
    for rel in relations:
        examples.extend(generate_training_examples(rel))
    return examples


def write_bert_examples(examples, path):
    """Cache expanded flat examples for a split to disk.

    Optional: `generate_training_examples` can be slow to redo on every run,
    so this lets you write out e.g. `train_bert_examples.json` once and
    reload it directly in later runs via `load_split` + `RelationDataset`.
    """
    with open(path, "w") as f:
        json.dump(examples, f, indent=2)


def make_hf_dataset(examples):
    """Wrap a list of {"text", "label"} dicts into a single HF `Dataset`."""
    return Dataset.from_dict({
        "text": [ex["text"] for ex in examples],
        "label": [ex["label"] for ex in examples],
    })


def tokenize_dataset(dataset, tokenizer, label2id, max_length=512):
    """Tokenize a raw (text, label) `Dataset` for classification training.

    Converts string labels to ints via `label2id`, leaving padding to the
    data collator at training time.
    """

    def preprocess_function(batch):
        encodings = tokenizer(
            batch["text"],
            truncation=False,
        )
    
        keep_indices = [
            i for i, ids in enumerate(encodings["input_ids"])
            if len(ids) <= max_length
        ]
    
        filtered = {
            key: [values[i] for i in keep_indices]
            for key, values in encodings.items()
        }
        filtered["labels"] = [label2id[batch["label"][i]] for i in keep_indices]
    
        return filtered

    return dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["text", "label"],
    )


def prepare_bert_dataset(split_paths, tokenizer, label2id, max_length=512):
    """Run the BERT-specific pipeline: load split files -> expand each raw
    nested relation into flat text/label examples -> tokenize.

    `split_paths` is the dict of split name -> file path returned by
    `data.splitting.prepare_data_splits` (or `write_splits`). Returns a
    tokenized `DatasetDict`, ready to hand to `Trainer`.
    """
    dataset_dict = {}
    for name, path in split_paths.items():
        relations = load_split(path)
        examples = build_bert_examples(relations)
        print(f"{name}: {len(relations)} relations -> {len(examples)} examples")
        dataset_dict[name] = tokenize_dataset(make_hf_dataset(examples), tokenizer, label2id, max_length)
        print(f"{name}: dataset_dict length = {len(dataset_dict[name])}")
    return DatasetDict(dataset_dict)