import itertools


class Node:
    """A node in a nested-relation tree.

    Either a leaf `entity` node (a plain entity mention), or a `relation`
    node whose `children` are `(arg_name, child_node)` pairs.
    """

    def __init__(self, kind, label):
        self.kind = kind
        self.label = label
        self.children = []
        self.level = 0
        self.parent_arg = None
        self.discovered = False

    def flat_text(self):
        if self.kind == "entity":
            return self.label
        inner = " ".join(f"[{a.upper()}]{c.flat_text()}[/{a.upper()}]" for a, c in self.children)
        tag = self.label.upper()
        return f"[{tag}]{inner}[/{tag}]"

    def output_text(self):
        if self.kind == 'entity':
            return self.label
        else:
            inner = ", ". join(f"{a}={c.output_text()}" for a, c in self.children)
            return f"{self.label}({inner})"


def parse_relation(rel):
    """Recursively build a `Node` tree from the nested relation-json structure."""
    if rel["type"] == "entity":
        return Node("entity", rel["entity"])

    node = Node("relation", rel["label"])
    for arg_name, child_json in rel["arguments"].items():
        child = parse_relation(child_json)
        node.children.append((arg_name, child))
        child.parent_arg = (node, arg_name)
    return node


def compute_levels(node):
    """Assign `node.level`: 0 for entities, 1 + max(child level) for relations."""
    if node.kind == "entity":
        node.level = 0
    else:
        node.level = 1 + max(compute_levels(c) for _, c in node.children)
    return node.level


def collect_all(node, acc=None):
    """Flatten the tree into a list of all nodes (pre-order)."""
    if acc is None:
        acc = []
    acc.append(node)
    for _, c in node.children:
        collect_all(c, acc)
    return acc


def get_nodes_by_level(nodes):
    """Group nodes by their `level` into {level: [nodes]}."""
    level_dict = {}
    for node in nodes:
        level_dict.setdefault(node.level, []).append(node)
    return level_dict


def render_context_segment(node, tag=None):
    """Render a node's flattened text, optionally wrapped in an [TAG]...[/TAG] marker."""
    text = node.flat_text()
    return f"[{tag}]{text}[/{tag}]" if tag else text


def generate_training_examples(rel_json):
    """Turn one nested-relation example into flat BERT training examples.

    Walks the relation tree round-by-round: in each round, every
    not-yet-discovered entity/relation in the current `pool` is offered up
    as a candidate (marked `[E1]`), and every ordered pair is offered up as
    a candidate pair (marked `[E1]`/`[E2]`), each labelled with the relation
    it resolves to (or "none" if it doesn't complete a relation yet).
    Resolved relations get folded into the pool for the next round, so
    nested relations are built up bottom-up.

    Returns a list of `{"text": ..., "label": ...}` dicts.
    """
    text = rel_json["sentence"]

    root = parse_relation(rel_json["relation"])
    compute_levels(root)
    all_nodes = collect_all(root)
    entities = [n for n in all_nodes if n.kind == "entity"]

    nodes_by_level = get_nodes_by_level(all_nodes)
    max_level = max(nodes_by_level.keys())
    max_rounds = max_level + 1

    training_examples = []
    discovered = list(entities)
    pool = list(entities)

    for _ in range(max_rounds):
        new_this_round = []

        for candidate in pool:
            parent_arg = candidate.parent_arg
            if parent_arg and parent_arg[0].discovered:
                continue

            label = "none"
            if parent_arg:
                parent, arg_name = parent_arg
                if len(parent.children) == 1 and not parent.discovered:
                    label = parent.label

            ctx_parts = [render_context_segment(n, "E1" if n is candidate else None) for n in discovered]
            new_text = text + " [SEP]" + "[SEP]".join(ctx_parts)

            training_examples.append({"text": new_text, "label": label})
            if label != "none":
                new_this_round.append(parent_arg[0])

        for c1, c2 in itertools.permutations(pool, 2):
            label = "none"
            if c1.parent_arg and c2.parent_arg and c1.parent_arg[0] is c2.parent_arg[0]:
                parent = c1.parent_arg[0]
                if len(parent.children) == 2 and not parent.discovered:
                    (a1, ch1), (a2, ch2) = parent.children
                    if ch1 is c1 and ch2 is c2:
                        label = parent.label

            ctx_marks = {c1: "E1", c2: "E2"}
            ctx_parts = [render_context_segment(n, ctx_marks.get(n)) for n in discovered]
            new_text = text + " [SEP]" + "[SEP]".join(ctx_parts)

            training_examples.append({"text": new_text, "label": label})
            if label != "none":
                p = c1.parent_arg[0]
                if p not in new_this_round:
                    new_this_round.append(p)

        if not new_this_round:
            break

        for parent in new_this_round:
            parent.discovered = True
            discovered.append(parent)
            pool.append(parent)

    return training_examples