import torch.distributions as dist
from typing import List, Dict
import itertools

start_token = "<|startoftext|>"
end_token = "<|endoftext|>"

def entropy_regularization(attention_maps, modifier, noun):
    entropy = 0
    
    for i in range(len(attention_maps)):
        if len(attention_maps[i].shape) > 1:
            attention_maps[i] = attention_maps[i].reshape(-1)
            attention_maps[i] = attention_maps[i].to(torch.float32)
            attention_maps[i] = torch.clamp(attention_maps[i], min=0)
            attention_maps[i] = attention_maps[i] / torch.sum(attention_maps[i])

    if isinstance(noun, list):
        for n in noun:
            entropy += dist.Categorical(probs=attention_maps[n]).entropy()
    else:
        entropy = dist.Categorical(probs=attention_maps[noun]).entropy()
    if isinstance(modifier, list):
        for m in modifier:
            entropy += dist.Categorical(probs=attention_maps[m]).entropy()
    else:
        entropy += dist.Categorical(probs=attention_maps[modifier]).entropy()
    # print("entropy:", entropy)
    return entropy

def _get_outside_indices(subtree_indices, attn_map_idx_to_wp):
    flattened_subtree_indices = _flatten_indices(subtree_indices)
    outside_indices = [
        map_idx
        for map_idx in attn_map_idx_to_wp.keys() if (map_idx not in flattened_subtree_indices)
    ]
    return outside_indices


def _flatten_indices(related_indices):
    flattened_related_indices = []
    for item in related_indices:
        if isinstance(item, list):
            flattened_related_indices.extend(item)
        else:
            flattened_related_indices.append(item)
    return flattened_related_indices


def split_indices(related_indices: List[int]):
    noun = [related_indices[-1]]  # assumes noun is always last in the list
    modifier = related_indices[:-1]
    if isinstance(modifier, int):
        modifier = [modifier]
    return noun, modifier



def calculate_positive_pairs(attention_maps, modifier, noun):
    src_indices = modifier
    dest_indices = noun

    if isinstance(src_indices, list) and isinstance(dest_indices, list):
        positive_pairs = [(attention_maps[s], attention_maps[d])
            for (s, d) in itertools.product(src_indices, dest_indices)
        ]
    elif isinstance(dest_indices, list):
        positive_pairs = [(attention_maps[src_indices], attention_maps[d])
            for d in dest_indices
        ]
    elif isinstance(src_indices, list):
        positive_pairs = [(attention_maps[s], attention_maps[dest_indices])
            for s in src_indices
        ]
    else:
        positive_pairs = [(attention_maps[src_indices], attention_maps[dest_indices])]
    
    return positive_pairs


def _calculate_outside_pairs(attention_maps, src_indices, outside_loss):
    negative_pairs = []
    computed_pairs = set()
    pair_counter = 0

    for outside_idx in outside_loss:
        if isinstance(src_indices, list):
            wp_neg_pairs= []
            for t in src_indices:
                pair_key = (t, outside_idx)
                
                if pair_key not in computed_pairs:
                    wp_neg_pairs.append((attention_maps[t], attention_maps[outside_idx]))
                    computed_pairs.add(pair_key)
            negative_pairs.append(wp_neg_pairs if wp_neg_pairs else None)
            pair_counter += 1

        else:
            pair_key = (src_indices, outside_idx)
            
            if pair_key not in computed_pairs:
                
                negative_pairs.append((attention_maps[src_indices], attention_maps[outside_idx]))
                computed_pairs.add(pair_key)
                pair_counter += 1
                
    return negative_pairs, pair_counter

def align_wordpieces_indices(
        wordpieces2indices, start_idx, target_word
):
    """
    Aligns a `target_word` that contains more than one wordpiece (the first wordpiece is `start_idx`)
    """

    wp_indices = [start_idx]
    wp = wordpieces2indices[start_idx].replace("</w>", "")

    # Run over the next wordpieces in the sequence (which is why we use +1)
    for wp_idx in range(start_idx + 1, len(wordpieces2indices)):
        if wp.lower() == target_word.lower():
            break

        wp2 = wordpieces2indices[wp_idx].replace("</w>", "")
        if target_word.lower().startswith(wp.lower() + wp2.lower()) and wp2.lower() != target_word.lower():
            wp += wordpieces2indices[wp_idx].replace("</w>", "")
            wp_indices.append(wp_idx)
        else:
            wp_indices = (
                []
            )  # if there's no match, you want to clear the list and finish
            break

    return wp_indices


def extract_attribution_indices(doc):
    # doc = parser(prompt)
    subtrees = []
    modifiers = ["amod", "nmod", "compound", "npadvmod", "advmod", "acomp"]

    for w in doc:
        if w.pos_ not in ["NOUN", "PROPN"] or w.dep_ in modifiers:
            continue
        subtree = []
        stack = []
        for child in w.children:
            if child.dep_ in modifiers:
                subtree.append(child)
                stack.extend(child.children)

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == "conj":
                subtree.append(node)
                stack.extend(node.children)
        if subtree:
            subtree.append(w)
            subtrees.append(subtree)
    return subtrees

def extract_attribution_indices_with_verbs(doc):
    '''This function specifically addresses cases where a verb is between
       a noun and its modifier. For instance: "a dog that is red"
       here, the aux is between 'dog' and 'red'. '''

    subtrees = []
    modifiers = ["amod", "nmod", "compound", "npadvmod", "advmod", "acomp",
                 'relcl']
    for w in doc:
        if w.pos_ not in ["NOUN", "PROPN"] or w.dep_ in modifiers:
            continue
        subtree = []
        stack = []
        for child in w.children:
            if child.dep_ in modifiers:
                if child.pos_ not in ['AUX', 'VERB']:
                    subtree.append(child)
                stack.extend(child.children)

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == "conj":
                # we don't want to add 'is' or other verbs to the loss, we want their children
                if node.pos_ not in ['AUX', 'VERB']:
                    subtree.append(node)
                stack.extend(node.children)
        if subtree:
            subtree.append(w)
            subtrees.append(subtree)
        return subtrees

def extract_attribution_indices_with_verb_root(doc):
    '''This function specifically addresses cases where a verb is between
       a noun and its modifier. For instance: "a dog that is red"
       here, the aux is between 'dog' and 'red'. '''

    subtrees = []
    modifiers = ["amod", "nmod", "compound", "npadvmod", "advmod", "acomp"]

    for w in doc:
        subtree = []
        stack = []

        # if w is a verb/aux and has a noun child and a modifier child, add them to the stack
        if w.pos_ != 'AUX' or w.dep_ in modifiers:
            continue

        for child in w.children:
            if child.dep_ in modifiers or child.pos_ in ['NOUN', 'PROPN']:
                if child.pos_ not in ['AUX', 'VERB']:
                    subtree.append(child)
                stack.extend(child.children)
        # did not find a pair of noun and modifier
        if len(subtree) < 2:
            continue

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == "conj":
                # we don't want to add 'is' or other verbs to the loss, we want their children
                if node.pos_ not in ['AUX']:
                    subtree.append(node)
                stack.extend(node.children)

        if subtree:
            if w.pos_ not in ['AUX']:
                subtree.append(w)
            subtrees.append(subtree)
    return subtrees


def extract_entities_only(doc):
    entities = []
    for w in doc:
        if w.pos_ in ['NOUN', 'PROPN']:
            entities.append([w])
    return entities

def calculate_negative_pairs(
        attention_maps, modifier, noun, subtree_indices, attn_map_idx_to_wp
):
    
    outside_indices = _get_outside_indices(subtree_indices, attn_map_idx_to_wp)
    # print(outside_indices)
    negative_noun_pairs, num_noun_pairs = _calculate_outside_pairs(
            attention_maps, noun, outside_indices
        )
    if outside_indices:
        negative_noun_pairs = negative_noun_pairs
    else:
        negative_pairs = []

    if modifier:
        negative_modifier_pairs, num_modifier_pairs = _calculate_outside_pairs(
                attention_maps, noun, outside_indices
            )
        if outside_indices:
            negative_modifier_pairs = negative_modifier_pairs
        else:
            negative_modifier_pairs = []

        negative_pairs = negative_modifier_pairs + negative_noun_pairs
    else:
        negative_pairs = negative_noun_pairs
    # print(f"len neg pairs:{len(negative_pairs)}, num: {num_noun_pairs}")
    return negative_pairs

def get_indices(tokenizer, prompt: str) -> Dict[str, int]:
    """Utility function to list the indices of the tokens you wish to alter"""
    ids = tokenizer(prompt).input_ids
    indices = {
        i: tok
        for tok, i in zip(
            tokenizer.convert_ids_to_tokens(ids), range(len(ids))
        )
    }
    return indices

def get_attention_map_index_to_wordpiece(tokenizer, prompt):
    attn_map_idx_to_wp = {}

    wordpieces2indices = get_indices(tokenizer, prompt)

    # Ignore `start_token` and `end_token`
    for i in list(wordpieces2indices.keys())[1:-1]:
        wordpiece = wordpieces2indices[i]
        wordpiece = wordpiece.replace("</w>", "")
        attn_map_idx_to_wp[i] = wordpiece
    # print(attn_map_idx_to_wp)
    return attn_map_idx_to_wp