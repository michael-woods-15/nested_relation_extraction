import os

# --- Paths -------------------------------------------------------------
# Raw data lives in its own top-level `resources/` directory, kept separate
# from the `data/` *package* (which holds data-handling code, not data files).
RESOURCES_DIR = "resources"

DATASET_PATH = os.path.join(RESOURCES_DIR, "nested_relations_dataset_v2.json")
SCHEMA_PATH = os.path.join(RESOURCES_DIR, "relation_schema.yml")
TERMS_PATH = os.path.join(RESOURCES_DIR, "terms.json")

# --- Model -----------------------------------------------------------------
MODEL_NAME = "t5-small"

# --- Data splitting / reproducibility --------------------------------------
SEED = 42
TRAIN_FRAC = 0.8   # fraction of data used for training
VAL_FRAC = 0.9     # cumulative fraction: train+val, remainder is test

# --- Training hyperparameters -----------------------------------------------
OUTPUT_DIR = "./bert-relation-extraction"
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
NUM_EPOCHS = 10
LOGGING_STEPS = 50
METRIC_FOR_BEST_MODEL = "macro_f1"
GREATER_IS_BETTER = True


# --- Relation Types for Labels ----------------------------------------------
ONE_ARG_RELS = [
    'increase',
    'decrease',
    'rna_expression',
    'protein_expression',
    'expression',
    'amplification',
    #'deletion',
    #'mutation',
    'production',
    'bioactivity',
    'phosphorylation',
    'phosphorylated',
    'dephosphorylation',
    'dephosphorylated',
    'ubiquitination',
    'deubiquitination',
    'methylation',
    'methylated',
    'demethylation',
    'demethylated',
]

TWO_ARG_RELS = [
    'effect',
    'correlation',
    #'binding'
]