import numpy as np
 
from metrics.bin_decomp import get_bin_rels, get_parse_fail_FN, binary_decomposition_metrics
from metrics.parser import parse_tree
from metrics.schema_checker import validate_predicate
from metrics.ted import mean_norm_tree_edit_distance

def evaluate_bert(preds, targets, relation_schema, terms, root_labels):
    """
    Parse, validate, and score a batch of predicted string lists vs. target strings.
        
    Returns a dict with `parse_rate`, `schema_rate` and the binary decomposition
    precision/recall/F1.
    """
    n = 0
    valid_parse_count = 0
    valid_schema_count = 0

    valid_preds = []
    valid_targets = []
    invalid_parse_targets = []

    for p, t in zip(preds, targets):
        parsed_target, _ = parse_tree(t)
        parsed_target = parsed_target.children[0]

        valid_parses = []
        n += len(p)
        for pred in p:
            parsed_pred, err = parse_tree(pred)
            if err is None:
                parsed_pred = parsed_pred.children[0]
                valid_parse_count += 1
    
                #Schema check
                ok, err = validate_predicate(parsed_pred, root_labels, relation_schema, terms)
                if ok:
                    valid_schema_count += 1

                valid_parses.append(parsed_pred)

        if valid_parses:
            valid_preds.append(valid_parses)
            valid_targets.append(parsed_target)
        else:
            invalid_parse_targets.append(parsed_target)

    parse_rate = valid_parse_count / n
    schema_rate = valid_schema_count / n

    pred_bin_rels, target_bin_rels = get_bin_rels(valid_preds, valid_targets)
    parse_fail_FN = get_parse_fail_FN(invalid_parse_targets)
    bin_decomp_metrics = binary_decomposition_metrics(pred_bin_rels, target_bin_rels, parse_fail_FN)

    return {
        "parse_rate": parse_rate,
        "schema_rate": schema_rate,
        **bin_decomp_metrics,

    }        
 

def evaluate_t5(preds, targets, relation_schema, terms, root_labels):
    """Parse, validate, and score a batch of predicted vs. target strings.
     
        Returns a dict with `parse_rate`, `schema_rate`, the binary decomposition
        precision/recall/F1, and `mean_norm_ted`.
    """
    n = len(preds)
    valid_parse_count = 0
    valid_schema_count = 0

    valid_preds = []
    valid_targets = []
    invalid_parse_targets = []

    for p, t in zip(preds, targets):
        parsed_target, _ = parse_tree(t)
        parsed_target = parsed_target.children[0]
        
        parsed_pred, err = parse_tree(p)
        if err is None:
            parsed_pred = parsed_pred.children[0]
            valid_parse_count += 1

            #Schema check
            ok, err = validate_predicate(parsed_pred, root_labels, relation_schema, terms)
            if ok:
                valid_schema_count += 1

            valid_preds.append(parsed_pred)
            valid_targets.append(parsed_target)
        else:
            invalid_parse_targets.append(parsed_target)

    parse_rate = valid_parse_count / n
    schema_rate = valid_schema_count / n

    pred_bin_rels, target_bin_rels = get_bin_rels(valid_preds, valid_targets)
    parse_fail_FN = get_parse_fail_FN(invalid_parse_targets)
    bin_decomp_metrics = binary_decomposition_metrics(pred_bin_rels, target_bin_rels, parse_fail_FN)

    mean_norm_ted = mean_norm_tree_edit_distance(valid_preds, valid_targets)

    return {
        "parse_rate": parse_rate,
        "schema_rate": schema_rate,
        **bin_decomp_metrics,
        "mean_norm_ted": mean_norm_ted

    }
 

def make_compute_metrics(tokenizer, relation_schema, terms, root_labels):
    """Build a `compute_metrics(eval_preds)` callable for `Seq2SeqTrainer`.
 
    Bundling `tokenizer`/`relation_schema`/`terms`/`root_labels` via closure
    (instead of relying on module-level globals, as the original notebook
    did) keeps this reusable across different tokenizers/schemas/runs.
    """
 
    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # Replace -100 in preds and labels (padding token used by data collator) before decoding
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        return evaluate_t5(decoded_preds, decoded_labels, relation_schema, terms, root_labels)
 
    return compute_metrics