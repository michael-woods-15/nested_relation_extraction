from itertools import count
 
from lark import Tree
 
from metrics.schema_checker import strip_quote_markers
 
 
def decompose(predicate_tree, binary_rels=None, ids=None, current=None):
    """Flatten a predicate tree into a set of binary (parent, arg, child) relations."""
    if binary_rels is None:
        binary_rels = []
        ids = count()
 
    if current is None:
        current = (next(ids), predicate_tree.children[0].value)
 
    for arg_node in predicate_tree.children[1:]:
        arg_name = arg_node.children[0].value
        inner = arg_node.children[1].children[0]
 
        if isinstance(inner, Tree) and inner.data == "predicate":
            child = (next(ids), inner.children[0].value)
            binary_rels.append((current, arg_name, child))
            decompose(inner, binary_rels, ids, child)
        else:
            entity = strip_quote_markers(str(inner))
            binary_rels.append((current, arg_name, entity))
 
    return set(binary_rels)
 
 
def bin_decompostion_metrics(pred_trees, target_trees):
    """Compute precision/recall/F1 over binary-decomposed predicate trees."""
    TP = 0
    FP = 0
    FN = 0
 
    for pred_tree, target_tree in zip(pred_trees, target_trees):
        pred_bin_rels = decompose(pred_tree)
        target_bin_rels = decompose(target_tree)
 
        TP += len(target_bin_rels & pred_bin_rels)
        FP += len(pred_bin_rels.difference(target_bin_rels))
        FN += len(target_bin_rels.difference(pred_bin_rels))
 
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
 
    return {
        "bin_decomp_precision": precision,
        "bin_decomp_recall": recall,
        "bin_decomp_f1_score": f1_score,
    }