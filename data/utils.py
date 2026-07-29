def get_labels():
    all_relation_labels = ONE_ARG_RELS + TWO_ARG_RELS + ["none"]
    label2id = {label: i for i, label in enumerate(all_relation_labels)}
    id2label = {i: label for label, i in label2id.items()}

    return label2id, id2label

def get_num_labels():
    label2id, _ = get_labels()
    return len(label2id)

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