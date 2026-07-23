import json
from datetime import datetime
from pathlib import Path
 
import optuna
import torch
from optuna.trial import TrialState
from transformers import TrainerCallback
 
import config
from data.dataset import load_and_prepare_datasets
from metrics.evaluate import make_compute_metrics
from training.t5_model import load_tokenizer_and_model
from training.t5_train import build_trainer
 
SEARCH_EPOCHS = 5

 
class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial, metric_key):
        self.trial = trial
        self.metric_key = metric_key
 
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.metric_key not in metrics:
            return control
        self.trial.report(metrics[self.metric_key], step=int(state.epoch))
        if self.trial.should_prune():
            control.should_training_stop = True
        return control

class OptunaHyperparameterSearch:
    def __init__(self, model_name, n_trials, seed=42):
        self.model_name = model_name
        self.safe_model_name = model_name.replace("/", ",")
        self.n_trials = n_trials
        self.seed = seed
        self.metric_key = f"eval_{config.METRIC_FOR_BEST_MODEL}"

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.study_name = f'{self.safe_model_name}_optimization_{timestamp}'
        storage_name = f"sqlite:///{self.safe_model_name}_optuna_study.db"

        self.tokenizer, _ = load_tokenizer_and_model(self.model_name)
 
        self.tokenized_datasets = load_and_prepare_datasets(
            config.DATASET_PATH,
            self.tokenizer,
            config.TASK_PREFIX,
            config.MAX_LENGTH,
            config.SEED,
            config.TRAIN_FRAC,
            config.VAL_FRAC,
        )
 
        self.compute_metrics = make_compute_metrics(
            self.tokenizer,
            config.SCHEMA_PATH,
            config.TERMS_PATH,
            config.ROOT_LABELS,
        )

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,  
        )
        
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=storage_name,
            direction='maximize',
            pruner=pruner,
            load_if_exists=True
        )

    def objective(self, trial):
        trial_lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        trial_batch_size = trial.suggest_categorical("per_device_batch_size", [4, 8, 16, 32])
 
        print(f"\nModel: {self.model_name} - Trial {trial.number} "
              f"(lr={trial_lr:.2e}, batch_size={trial_batch_size})")
 
        _, model = load_tokenizer_and_model(self.model_name)
        pruning_callback = OptunaPruningCallback(trial, self.metric_key)

        try:
            trainer = build_trainer(
                model,
                self.tokenizer,
                self.tokenized_datasets,
                self.compute_metrics,
                output_dir=config.OUTPUT_DIR,
                learning_rate=trial_lr,
                train_batch_size=trial_batch_size,
                eval_batch_size=trial_batch_size,
                num_epochs=config.NUM_EPOCHS,
                generation_max_length=config.GENERATION_MAX_LENGTH,
                logging_steps=config.LOGGING_STEPS,
                metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
                greater_is_better=config.GREATER_IS_BETTER,
                callbacks=[pruning_callback]
            )
        
            trainer.train()
            eval_metrics = trainer.evaluate()
 
        except torch.cuda.OutOfMemoryError:
            print(f"Trial {trial.number} ran out of memory at batch_size={trial_batch_size}")
            raise optuna.TrialPruned()

        return eval_metrics[self.metric_key]
    
    def run_search(self):
        print(f"\n{'='*80}")
        print("Starting Optuna Hyperparameter Search")
        print(f"Model Name: {self.model_name}")
        print(f"Random seed: {self.seed}")
        print(f"\n{'='*80}")

        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            catch=(Exception,)
        )

        self.save_results()

    def save_results(self):
        study_stats = {
            'n_trials': len(self.study.trials),
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'best_trial_number': self.study.best_trial.number,
            'seed': self.seed,
            'datetime_start': self.study.trials[0].datetime_start.isoformat() if self.study.trials else None,
            'datetime_complete': datetime.now().isoformat()
        }
        
        with open(f'{self.safe_model_name}_study_summary.json', 'w') as f:
            json.dump(study_stats, f, indent=2)

        trials_data = []
        for trial in self.study.trials:
            if trial.state == TrialState.COMPLETE:
                trial_dict = {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'user_attrs': trial.user_attrs,
                    'state': trial.state.name
                }
                trials_data.append(trial_dict)
        
        with open(f'{self.safe_model_name}_all_trials.json', 'w') as f:
            json.dump(trials_data, f, indent=2)


        

if __name__ == '__main__':
    MODELS = [
        "t5-small",
        "t5-base",
        "razent/SciFive-base-Pubmed_PMC"
    ]

    for model in MODELS:
        searcher = OptunaHyperparameterSearch(model, n_trials=10)
        searcher.run_search()