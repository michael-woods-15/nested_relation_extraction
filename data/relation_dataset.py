from torch.utils.data import Dataset

class RelationDataset(Dataset):
    def __init__(self, examples, tokenizer, label2id, max_length=256):
        self.labels = [label2id[ex["label"]] for ex in examples]
        self.encodings = tokenizer(
            [ex["text"] for ex in examples],
            truncation=True,
            max_length=max_length,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item