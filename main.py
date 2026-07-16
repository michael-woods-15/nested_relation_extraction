import json
from datetime import datetime

import config
from data.dataset import load_and_prepare_datasets
from metrics.evaluate import make_compute_metrics
from training.t5_model import load_tokenizer_and_model
from training.t5_train import build_trainer
 
 
def main():
    tokenizer, model = load_tokenizer_and_model(config.MODEL_NAME)
 
    tokenized_datasets = load_and_prepare_datasets(
        config.DATASET_PATH,
        tokenizer,
        config.TASK_PREFIX,
        config.MAX_LENGTH,
        config.SEED,
        config.TRAIN_FRAC,
        config.VAL_FRAC,
    )
 
    compute_metrics = make_compute_metrics(
        tokenizer,
        config.SCHEMA_PATH,
        config.TERMS_PATH,
        config.ROOT_LABELS,
    )
 
    trainer = build_trainer(
        model,
        tokenizer,
        tokenized_datasets,
        compute_metrics,
        output_dir=config.OUTPUT_DIR,
        learning_rate=config.LEARNING_RATE,
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

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f"results/{config.MODEL_NAME}_{timestamp}", "w") as f:
        json.dump(output, f, indent=2)
 
 
if __name__ == "__main__":
    main()