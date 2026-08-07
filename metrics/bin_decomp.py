from lark import Tree
 
from metrics.schema_checker import strip_quote_markers
 
 
def decompose(predicate_tree):
    def _decompose(node):
        pred_name = node.children[0].value
        rels = set()
        children_repr = []

        for arg_node in node.children[1:]:
            arg_name = arg_node.children[0].value
            inner = arg_node.children[1].children[0]

            if isinstance(inner, Tree) and inner.data == "predicate":
                child_id, child_rels = _decompose(inner)
                rels |= child_rels
                children_repr.append((arg_name, child_id))
            else:
                entity = strip_quote_markers(str(inner))
                children_repr.append((arg_name, entity))

        current = (hash((pred_name, tuple(children_repr))), pred_name)

        for arg_name, child in children_repr:
            rels.add((current, arg_name, child))

        return current, rels

    _, binary_rels = _decompose(predicate_tree)
    return binary_rels
 
 
def get_bin_rels(valid_preds, valid_targets):
    all_pred_bin_rels = []
    all_target_bin_rels = []

    for p, t in zip(valid_preds, valid_targets):
        if isinstance(p, list):
            pred_bin_rels = set().union(*(decompose(pred) for pred in p))
        else:
            pred_bin_rels = decompose(p)
            
        target_bin_rels = decompose(t)

        all_pred_bin_rels.append(pred_bin_rels)
        all_target_bin_rels.append(target_bin_rels)

    return all_pred_bin_rels, all_target_bin_rels


def get_parse_fail_FN(invalid_targets):
    FN = 0
    for t in invalid_targets:
        bin_rels = decompose(t)
        FN += len(bin_rels)

    return FN


def binary_decomposition_metrics(pred_bin_rels, target_bin_rels, parse_fail_FN):
    TP = 0
    FP = 0
    FN = parse_fail_FN

    for p_bin_rels, t_bin_rels in zip(pred_bin_rels, target_bin_rels):
        TP += len(t_bin_rels & p_bin_rels)
        FP += len(p_bin_rels.difference(t_bin_rels))
        FN += len(t_bin_rels.difference(p_bin_rels))

    precision = TP / (TP+FP) if (TP+FP) else 0
    recall = TP / (TP+FN) if (TP+FN) else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "bin_decomp_precision" : precision,
        "bin_decomp_recall" : recall,
        "bin_decomp_f1_score" : f1_score,
    }