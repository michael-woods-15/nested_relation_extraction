import json
from datetime import datetime
import os

import config_t5 as config
from data.t5_dataset import load_and_prepare_datasets
from data.utils import prepare_data_splits
from metrics.eval_metrics import make_compute_metrics
from metrics.schema_checker import load_relation_schema, load_terms
from training.t5_model import load_tokenizer_and_model
from training.t5_train import build_trainer
 
 
def main():
    tokenizer, model = load_tokenizer_and_model(config.MODEL_NAME)

    splits, split_paths = prepare_data_splits(
        config.DATASET_PATH,
        config.SPLITS_DIR,
        config.SEED,
        config.TRAIN_FRAC,
        config.VAL_FRAC
    )
 
    tokenized_datasets = load_and_prepare_datasets(
        split_paths,
        tokenizer,
        config.TASK_PREFIX,
        config.MAX_LENGTH,
    )

    relation_schema = load_relation_schema(config.SCHEMA_PATH)
    terms = load_terms(config.TERMS_PATH)
 
    compute_metrics = make_compute_metrics(
        tokenizer,
        relation_schema,
        terms,
        config.ROOT_LABELS,
    )
 
    trainer = build_trainer(
        model,
        tokenizer,
        tokenized_datasets,
        compute_metrics,
        output_dir=config.OUTPUT_DIR,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        train_batch_size=config.TRAIN_BATCH_SIZE,
        eval_batch_size=config.EVAL_BATCH_SIZE,
        num_epochs=config.NUM_EPOCHS,
        generation_max_length=config.GENERATION_MAX_LENGTH,
        logging_steps=config.LOGGING_STEPS,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        greater_is_better=config.GREATER_IS_BETTER,
    )
 
    trainer.train()
    output = trainer.evaluate()

    os.makedirs("model_weights", exist_ok=True)
    safe_model_name = config.MODEL_NAME.replace("/", "-")
    trainer.save_model(f"model_weights/{safe_model_name}")
    tokenizer.save_pretrained(f"model_weights/{safe_model_name}")

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f"results/{config.MODEL_NAME}_{timestamp}.json", "w") as f:
        json.dump(output, f, indent=2)
 
 
if __name__ == "__main__":
    main()