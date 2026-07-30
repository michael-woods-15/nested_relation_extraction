import json
import os
import random
from config_bert import ONE_ARG_RELS, TWO_ARG_RELS


def load_relations(dataset_path):
    """Load the raw list of relation examples from the dataset json file."""
    with open(dataset_path) as f:
        return json.load(f)


def split_relations(relations, seed, train_frac, val_frac):
    """Shuffle and split raw relation dicts into train/validation/test lists.

    `train_frac` and `val_frac` are cumulative fractions of the full dataset
    (e.g. 0.8 / 0.9 gives an 80/10/10 train/val/test split). Splitting happens
    on the raw relation dicts (before any model-specific processing), so the
    same split is reused by every downstream model.
    """
    n = len(relations)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    shuffled = [relations[i] for i in indices]

    train_split = int(train_frac * n)
    val_split = int(val_frac * n)

    return {
        "train": shuffled[:train_split],
        "validation": shuffled[train_split:val_split],
        "test": shuffled[val_split:],
    }


def write_splits(splits, output_dir):
    """Write each split (a list of raw relation dicts) to its own json file.

    `filenames` maps split name -> filename; defaults to
    {"train": "train_rels.json", "validation": "val_rels.json", "test": "test_rels.json"}.

    Returns a dict of split name -> path written, which can be handed
    straight to either model's dataset-loading function.
    """
    filenames = {"train": "train_rels.json", "validation": "val_rels.json", "test": "test_rels.json"}
    os.makedirs(output_dir, exist_ok=True)

    paths = {}
    for name, examples in splits.items():
        path = os.path.join(output_dir, filenames[name])
        with open(path, "w") as f:
            json.dump(examples, f, indent=2)
        paths[name] = path

    return paths


def prepare_data_splits(dataset_path, output_dir, seed, train_frac, val_frac):
    """Full shared first stage: load raw json -> shuffle -> split -> write to disk.

    This is the common entry point for both the T5 and BERT pipelines. Once
    the split files exist, each model-specific pipeline loads them and
    handles them however it needs:
      - T5:   data.dataset.load_and_prepare_datasets
      - BERT: data.bert_dataset.prepare_bert_dataset

    Returns (splits, paths) where `splits` is the in-memory dict of raw
    relation lists and `paths` is the dict of split name -> file path.
    """
    relations = load_relations(dataset_path)
    splits = split_relations(relations, seed, train_frac, val_frac)
    paths = write_splits(splits, output_dir)
    return splits, paths


def get_labels():
    """
    Return label mappings for all different relattion types that can be outputted by BERT model
    Returns a mapping from integer id's to labels, and vice versa.
    """
    all_relation_labels = ONE_ARG_RELS + TWO_ARG_RELS + ["none"]
    label2id = {label: i for i, label in enumerate(all_relation_labels)}
    id2label = {i: label for label, i in label2id.items()}

    return label2id, id2label


def get_num_labels():
    """
    Return the total number of labels between 1-arg, 2-arg rels and 'none'
    """
    label2id, _ = get_labels()
    return len(label2id)