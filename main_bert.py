import json
from datetime import datetime
import os

import config_bert as config
from training.bert_classifier import load_tokenizer_and_classifier
from training.bert_train import build_trainer
from data.bert_dataset import prepare_bert_dataset
from data.utils import prepare_data_splits, get_labels

 
 
def main():
    tokenizer, classifier = load_tokenizer_and_classifier(config.MODEL_NAME)

    label2id, id2label = get_labels()

    splits, split_paths = prepare_data_splits(
        config.DATASET_PATH,
        config.SPLITS_DIR,
        config.SEED,
        config.TRAIN_FRAC,
        config.VAL_FRAC
    )
 
    tokenized_datasets = prepare_bert_dataset(
        split_paths,
        tokenizer,
        label2id,
        config.MAX_LENGTH
    )

    trainer = build_trainer(
        classifier,
        tokenizer,
        tokenized_datasets,
        output_dir=config.OUTPUT_DIR,
        epochs=config.NUM_EPOCHS,
        train_batch_size=config.TRAIN_BATCH_SIZE,
        eval_batch_size=config.EVAL_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        logging_steps=config.LOGGING_STEPS,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        greater_is_better=config.GREATER_IS_BETTER
    )
 
    trainer.train()
    output = trainer.evaluate()

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f"results/{config.MODEL_NAME}_{timestamp}.json", "w") as f:
        json.dump(output, f, indent=2)
 
 
if __name__ == "__main__":
    main()