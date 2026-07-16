from apted import APTED, Config
from lark import Tree
 
from metrics.schema_checker import strip_quote_markers
 
 
class TedNode:
    __slots__ = ("label", "children")
 
    def __init__(self, label, children=None):
        self.label = label
        self.children = children or []
 
    def __str__(self):
        return ted_node_to_str(self)
 
 
def ted_node_to_str(node, indent=0):
    pad = "  " * indent
    lines = [f"{pad}{node.label}"]
    for child in node.children:
        lines.append(ted_node_to_str(child, indent + 1))
    return "\n".join(lines)
 
 
def to_ted_tree(predicate_tree):
    """Convert a parsed predicate tree into a `TedNode` tree for APTED."""
    label = predicate_tree.children[0].value
    node = TedNode(label)
 
    for arg_node in predicate_tree.children[1:]:
        if arg_node is None:
            continue
        arg_name = arg_node.children[0].value
        inner = arg_node.children[1].children[0]
 
        role_node = TedNode(f"@{arg_name}")
        if isinstance(inner, Tree) and inner.data == "predicate":
            role_node.children = [to_ted_tree(inner)]
        else:
            entity = strip_quote_markers(str(inner))
            role_node.children = [TedNode(f"={entity}")]
 
        node.children.append(role_node)
 
    node.children.sort(key=lambda c: c.label)
    return node
 
 
class TedConfig(Config):
    def rename(self, node1, node2):
        return 0 if node1.label == node2.label else 1  # unit-cost relabel
 
    def children(self, node):
        return node.children
 
    def insert(self, node):
        return 1  # unit-cost insert
 
    def delete(self, node):
        return 1  # unit-cost delete
 
 
def tree_edit_distance(pred_tree, gold_tree):
    """Raw (unnormalized) edit distance between two predicate trees."""
    pred_ted = to_ted_tree(pred_tree)
    gold_ted = to_ted_tree(gold_tree)
    return APTED(pred_ted, gold_ted, TedConfig()).compute_edit_distance()
 
 
def tree_size(node):
    return 1 + sum(tree_size(c) for c in node.children)
 
 
def normalized_ted(pred_tree, gold_tree):
    """Edit distance normalized to a [0, 1] similarity score (1 = identical)."""
    pred_ted = to_ted_tree(pred_tree)
    gold_ted = to_ted_tree(gold_tree)
    dist = APTED(pred_ted, gold_ted, TedConfig()).compute_edit_distance()
    max_size = max(tree_size(pred_ted), tree_size(gold_ted))
    return 1 - dist / max_size
 
 
def mean_norm_tree_edit_distance(pred_trees, target_trees):
    """Mean normalized tree edit distance/similarity across a batch of pairs."""
    n = len(pred_trees)
    cum_norm_ted = 0
 
    for pred_tree, target_tree in zip(pred_trees, target_trees):
        cum_norm_ted += normalized_ted(pred_tree, target_tree)
 
    return cum_norm_ted / n if n else 0.0