import json
from functools import lru_cache
 
import yaml
from lark import Tree
 
 
@lru_cache(maxsize=None)
def load_terms(path):
    """Load and index entity term lists from `path`.
 
    Cached per-path since this is called on every evaluation step but the
    underlying file doesn't change during training.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
 
    terms = {category: set(entities) for category, entities in raw.items()}
    terms["biomolecule"] = terms["chemical"].union(terms["gene"])
    terms["gene_protein"] = terms["gene"]
    return terms
 
 
@lru_cache(maxsize=None)
def load_relation_schema(path):
    """Load the relation schema (allowed relations, required/typed args)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
 
 
def strip_quote_markers(string_token_text):
    """Strip the `[QUOTE]...[QUOTE]` wrapper used to mark string literals."""
    return string_token_text[len("[QUOTE]"):-len("[QUOTE]")]
 
 
def validate_predicate(predicate_tree, allowed_types, relation_schema, terms):
    """Recursively validate a predicate node against the schema.
 
    Returns `(is_valid, error_message)`, where `error_message` is None on
    success.
    """
    label = predicate_tree.children[0].value
    arg_nodes = [c for c in predicate_tree.children[1:] if c is not None]
 
    if label not in relation_schema:
        return False, f"unknownn_relation_label: '{label}' is not defined in relation_schema"
 
    if allowed_types is not None and label not in allowed_types:
        return False, (
            f"disallowed_relation_type: '{label}' is not permitted in this position "
            f"(allowed: {allowed_types})"
        )
 
    definition = relation_schema[label]
    required_args = set(definition["args"])
 
    present = {}
    for arg_node in arg_nodes:
        arg_name = arg_node.children[0].value
        arg_tree = arg_node.children[1]
        present[arg_name] = arg_tree
 
    present_names = set(present.keys())
 
    missing = required_args - present_names
    if missing:
        return False, f"missing_args: '{label}' is missing required arg(s) {sorted(missing)}"
 
    unexpected = present_names - required_args
    if unexpected:
        return False, f"unexpected_args: '{label}' has unrecognized arg(s) {sorted(unexpected)}"
 
    for arg_name in required_args:
        arg_tree = present[arg_name]
        arg_types = definition.get("arg_types", {}).get(arg_name, [arg_name])
        ok, err = validate_arg(arg_tree, arg_types, relation_schema, terms)
        if not ok:
            return False, f"in {label}.{arg_name}: {err}"
 
    return True, None
 
 
def validate_arg(arg_tree, allowed_types, relation_schema, terms):
    """Validate a single argument: either a nested predicate or a leaf entity."""
    inner = arg_tree.children[0]
 
    if isinstance(inner, Tree) and inner.data == "predicate":
        return validate_predicate(inner, allowed_types, relation_schema, terms)
 
    entity_str = strip_quote_markers(str(inner))
    entity_categories = [t for t in allowed_types if t in terms]
 
    for cat in entity_categories:
        if entity_str in terms[cat]:  # O(1): terms[cat] is a set
            return True, None
 
    if not entity_categories:
        return False, (
            f"unexpected_entity_position: got a bare entity '{entity_str}' but none of the "
            f"allowed types here are entity categories (allowed_types={allowed_types})"
        )
    return False, (
        f"unknown_entity: '{entity_str}' not found in any allowed category {entity_categories}"
    )