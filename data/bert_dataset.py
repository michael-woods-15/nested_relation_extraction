import json

from data.bert_data_creation import generate_training_examples
from data.relation_dataset import RelationDataset


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


def prepare_bert_dataset(split_paths, tokenizer, label2id, max_length=256):
    """Run the BERT-specific pipeline: load split files -> expand each raw
    nested relation into flat text/label examples -> wrap in `RelationDataset`.

    `split_paths` is the dict of split name -> file path returned by
    `data.splitting.prepare_data_splits` (or `write_splits`). Returns a dict
    of split name -> `RelationDataset`, ready to hand to `Trainer`.
    """
    datasets = {}
    for name, path in split_paths.items():
        relations = load_split(path)
        examples = build_bert_examples(relations)
        datasets[name] = RelationDataset(examples, tokenizer, label2id, max_length)
    return datasets