import os

# --- Paths -------------------------------------------------------------
# Raw data lives in its own top-level `resources/` directory, kept separate
# from the `data/` *package* (which holds data-handling code, not data files).
RESOURCES_DIR = "resources"

DATASET_PATH = os.path.join(RESOURCES_DIR, "nested_relations_dataset_v2.json")
SCHEMA_PATH = os.path.join(RESOURCES_DIR, "relation_schema.yml")
TERMS_PATH = os.path.join(RESOURCES_DIR, "terms.json")

# --- Task / labels -------------------------------------------------------
TASK_PREFIX = "extract relations: "
ROOT_LABELS = ["effect"]

# --- Model -----------------------------------------------------------------
MODEL_NAME = "t5-small"
MAX_LENGTH = 512

# --- Data splitting / reproducibility --------------------------------------
SEED = 42
TRAIN_FRAC = 0.8   # fraction of data used for training
VAL_FRAC = 0.9     # cumulative fraction: train+val, remainder is test