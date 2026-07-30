import re
 
 
def mark_entities(sentence, entity_list):
    """Wrap each entity mention in the sentence with `[E{n}] ... [/E{n}]` markers.
 
    Entities are matched longest-first (so a shorter entity name that's a
    substring of a longer one doesn't get marked first), and numbered by
    their first position of occurrence in the sentence.
    """
    search_order = sorted(entity_list, key=len, reverse=True)
 
    first_positions = {}
    for ent in search_order:
        match = re.search(rf"\b{re.escape(ent)}\b", sentence)
        if match:
            first_positions[ent] = match.start()
 
    ordered_entities = sorted(first_positions, key=first_positions.get)
    entity_to_label = {ent: i + 1 for i, ent in enumerate(ordered_entities)}
 
    marked = sentence
    for ent in search_order:
        if ent not in entity_to_label:
            continue
        label = entity_to_label[ent]
        pattern = rf"\b{re.escape(ent)}\b"
        marked = re.sub(pattern, f"[E{label}] {ent} [/E{label}]", marked)
 
    return marked
 
 
def build_input_text(sentence, entity_list, task_prefix):
    """Build the full model input: task prefix + entity-marked sentence + entity suffix."""
    marked = mark_entities(sentence, entity_list)
    entity_suffix = "".join([f"[SEP]{entity}" for entity in entity_list])
    return task_prefix + marked + entity_suffix