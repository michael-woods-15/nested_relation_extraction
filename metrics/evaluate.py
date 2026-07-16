import numpy as np
 
from metrics.bin_decomp import bin_decompostion_metrics
from metrics.parser import parse_tree
from metrics.schema_checker import load_relation_schema, load_terms, validate_predicate
from metrics.ted import mean_norm_tree_edit_distance
 
 
def parse_and_evaluate(preds, targets, schema_path, terms_path, root_labels):
    """Parse, validate, and score a batch of predicted vs. target strings.
 
    Returns a dict with `parse_rate`, `schema_rate`, the binary decomposition
    precision/recall/F1, and `mean_norm_ted`. Predictions that fail to parse
    or fail schema validation are excluded from the downstream metrics (but
    still count against `parse_rate` / `schema_rate`).
    """
    n = len(preds)
 
    parsed_targets = [parse_tree(t) for t in targets]
    parsed_trees = [parse_tree(p) for p in preds]
    parse_rate = sum(1 for _, err in parsed_trees if err is None) / n if n > 0 else 0.0
 
    valid_parse_preds_targets = [
        (pred_tree.children[0], target_tree.children[0])
        for (pred_tree, perr), (target_tree, terr) in zip(parsed_trees, parsed_targets)
        if perr is None and terr is None
    ]
 
    relation_schema = load_relation_schema(schema_path)
    terms = load_terms(terms_path)
 
    valid_schema_pred_trees = []
    valid_schema_target_trees = []
    n_after_parse = len(valid_parse_preds_targets)
    for pred_tree, target_tree in valid_parse_preds_targets:
        ok, _err = validate_predicate(pred_tree, root_labels, relation_schema, terms)
        if ok:
            valid_schema_pred_trees.append(pred_tree)
            valid_schema_target_trees.append(target_tree)
 
    schema_rate = len(valid_schema_pred_trees) / n_after_parse if n_after_parse > 0 else 0.0
 
    bin_decomp_results = bin_decompostion_metrics(valid_schema_pred_trees, valid_schema_target_trees)
    mean_norm_ted = mean_norm_tree_edit_distance(valid_schema_pred_trees, valid_schema_target_trees)
 
    return {
        "parse_rate": parse_rate,
        "schema_rate": schema_rate,
        **bin_decomp_results,
        "mean_norm_ted": mean_norm_ted,
    }
 

def make_compute_metrics(tokenizer, schema_path, terms_path, root_labels):
    """Build a `compute_metrics(eval_preds)` callable for `Seq2SeqTrainer`.
 
    Bundling `tokenizer`/`schema_path`/`terms_path`/`root_labels` via closure
    (instead of relying on module-level globals, as the original notebook
    did) keeps this reusable across different tokenizers/schemas/runs.
    """
 
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
 
        # Replace -100 (padding sentinel used by the data collator) before decoding.
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
 
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
 
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
 
        return parse_and_evaluate(decoded_preds, decoded_labels, schema_path, terms_path, root_labels)
 
    return compute_metrics