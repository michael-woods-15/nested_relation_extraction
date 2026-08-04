import itertools

from data.bert_data_creation import Node, render_context_segment, compute_levels


single_arg_names = {
    'increase': 'variable',
    'decrease': 'variable',
    'rna_expression': 'gene_protein',
    'protein_expression': 'gene_protein',
    'expression': 'gene_protein',
    'amplification': 'gene_protein',
    'deletion': 'gene_protein',
    'mutation': 'gene_protein',
    'production': 'biomolecule',
    'bioactivity': 'biomolecule',
    'phosphorylation': 'gene_protein',
    'phosphorylated': 'gene_protein',
    'dephosphorylation': 'gene_protein',
    'dephosphorylated': 'gene_protein',
    'ubiquitination': 'gene_protein',
    'deubiquitination': 'gene_protein',
    'methylation': 'gene_protein',
    'methylated': 'gene_protein',
    'demethylation': 'gene_protein',
    'demethylated': 'gene_protein',
}

two_arg_names = {
    'effect': ('cause', 'theme'),
    'correlation': ('cause', 'theme')
}


def predict_relation(text, model, tokenizer, id2label, device, max_length=512):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    pred_id = torch.argmax(logits, dim=1).item()
    return id2label[pred_id]


def create_entity_node(entity):
    return Node('entity', entity)


def bert_multipass_inference(sentence, entities, model, tokenizer, id2label, device, verbose=False):
    entity_nodes = [create_entity_node(e) for e in entities]
    discovered = list(entity_nodes)
    pool = list(entity_nodes)
    counter = 1
    
    while True:
        if verbose:
            print("="*40)
            print(f"\nRound {counter}")
            print("="*40)
            
        new_this_round = []
        discovered_this_round = []

        for candidate in pool:
            if candidate.discovered:
                continue
                
            ctx_parts = [render_context_segment(n, 'E1' if n is candidate else None) for n in discovered]
            new_text = sentence + " [SEP]" + "[SEP]".join(ctx_parts)

            label = predict_relation(new_text, model, tokenizer, id2label, device)
            if verbose:
                print(f"\n{new_text}")
                print(label)
            
            if label != 'none' and single_arg_names.get(label, None):
                new_node = Node('relation', label)
                arg_name = single_arg_names[label]
                new_node.children.append((arg_name, candidate))
                candidate.parent_arg = (new_node, arg_name)

                discovered_this_round.append(candidate)
                new_this_round.append(new_node)


        for c1, c2 in itertools.permutations(pool, 2):
            if c1.discovered or c2.discovered:
                continue

            ctx_marks = {c1: 'E1', c2: 'E2'}
            ctx_parts = [render_context_segment(n, ctx_marks.get(n)) for n in discovered]
            new_text = sentence + " [SEP]" + "[SEP]".join(ctx_parts)

            label = predict_relation(new_text, model, tokenizer, id2label, device)
            if verbose:
                print(f"\n{new_text}")
                print(label)

            if label != 'none' and two_arg_names.get(label, None):
                new_node = Node('relation', label)
                args = two_arg_names[label]
                new_node.children.append((args[0], c1))
                c1.parent_arg = (new_node, args[0])
                new_node.children.append((args[1], c2))
                c2.parent_arg = (new_node, args[1])

                if c1 not in discovered_this_round:
                    discovered_this_round.append(c1)
                if c2 not in discovered_this_round:
                    discovered_this_round.append(c2)
                new_this_round.append(new_node)
                
            

        for c in discovered_this_round:
            c.discovered = True
            
        if not new_this_round:
            break
        for parent in new_this_round:
            discovered.append(parent)
            pool.append(parent)

        counter += 1

    max_level = -1
    max_level_node = None
    for node in discovered:
        level = compute_levels(node)
        if level > max_level:
            max_level = level
            max_level_node = node


    return max_level_node


def multipass_inference(relations, model, tokenizer, id2label, device, verbose=False):
    results = []

    for rel in relations:
        output = bert_multipass_inference(
            sentence=rel['sentence'],
            entities=rel['entities'],
            model=model,
            tokenizer=tokenizer,
            id2label=id2label,
            device=device,
            verbose=verbose
        )

        results.append(output)

    return results