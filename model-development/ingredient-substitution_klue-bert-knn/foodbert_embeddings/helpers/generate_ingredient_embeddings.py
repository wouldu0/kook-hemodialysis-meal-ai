import json
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from foodbert.helpers.prediction_model import PredictionModel


def _generate_food_sentence_dict():
    with Path('foodbert_embeddings/data/used_ingredients_clean.json').open(encoding='utf-8') as f:
        food_items = json.load(f)
        food_items_set = set(food_items)

    with Path('foodbert/data/train_instructions.txt').open(encoding='utf-8') as f:
        train_instruction_sentences = f.read().splitlines()
        train_instruction_sentences = [s for s in train_instruction_sentences if len(s.split()) <= 100]

    with Path('foodbert/data/test_instructions.txt').open(encoding='utf-8') as f:
        test_instruction_sentences = f.read().splitlines()
        test_instruction_sentences = [s for s in test_instruction_sentences if len(s.split()) <= 100]

    instruction_sentences = train_instruction_sentences + test_instruction_sentences

    food_to_sentences_dict = defaultdict(list)
    for sentence in instruction_sentences:
        for food in food_items_set:
            if food in sentence:
                food_to_sentences_dict[food].append(sentence)

    return food_to_sentences_dict


def _random_sample_with_min_count(population, k):
    if len(population) <= k:
        return population
    else:
        return random.sample(population, k)


def sample_random_sentence_dict(max_sentence_count):
    food_to_sentences_dict = _generate_food_sentence_dict()
    food_to_sentences_dict_random_samples = {food: _random_sample_with_min_count(sentences, max_sentence_count) for
                                             food, sentences in food_to_sentences_dict.items()}
    return food_to_sentences_dict_random_samples


def _merge_synonmys(food_to_embeddings_dict, max_sentence_count):
    synonmy_replacements_path = Path('foodbert_embeddings/data/synonmy_replacements.json')
    if synonmy_replacements_path.exists():
        with synonmy_replacements_path.open(encoding='utf-8') as f:
            synonmy_replacements = json.load(f)
    else:
        synonmy_replacements = {}

    merged_dict = defaultdict(list)
    for key, value in food_to_embeddings_dict.items():
        key_to_use = synonmy_replacements.get(key, key)
        merged_dict[key_to_use].append(value)

    merged_dict = {k: np.concatenate(v) for k, v in merged_dict.items()}
    for key, value in merged_dict.items():
        if len(value) > max_sentence_count:
            index = np.random.choice(value.shape[0], max_sentence_count, replace=False)
            merged_dict[key] = value[index]

    return merged_dict


def generate_food_embedding_dict(max_sentence_count):
    food_to_embeddings_dict_path = Path('foodbert_embeddings/data/food_embeddings_dict_foodbert.pkl')
    if food_to_embeddings_dict_path.exists():
        with food_to_embeddings_dict_path.open('rb') as f:
            food_to_embeddings_dict = pickle.load(f)

        old_ingredients = set(food_to_embeddings_dict.keys())
        with Path('foodbert_embeddings/data/used_ingredients_clean.json').open(encoding='utf-8') as f:
            new_ingredients = set(json.load(f))

        keys_to_delete = old_ingredients.difference(new_ingredients)
        for key in keys_to_delete:
            food_to_embeddings_dict.pop(key, None)

        food_to_embeddings_dict = _merge_synonmys(food_to_embeddings_dict, max_sentence_count)

        with food_to_embeddings_dict_path.open('wb') as f:
            pickle.dump(food_to_embeddings_dict, f)

        return food_to_embeddings_dict

    print('Sampling Random Sentences')
    food_to_sentences_dict_random_samples = sample_random_sentence_dict(max_sentence_count=max_sentence_count)
    print(f'문장 매칭된 재료: {len(food_to_sentences_dict_random_samples)}개')

    food_to_embeddings_dict = defaultdict(list)
    prediction_model = PredictionModel()

    for food, sentences in tqdm(food_to_sentences_dict_random_samples.items(),
                                total=len(food_to_sentences_dict_random_samples),
                                desc='Calculating Embeddings for Food items'):
        for sentence in sentences:
            embedding = prediction_model.get_ingredient_embedding(sentence, food)
            if embedding is not None:
                food_to_embeddings_dict[food].append(embedding)

    food_to_embeddings_dict = {k: np.stack(v) for k, v in food_to_embeddings_dict.items() if len(v) > 0}
    food_to_embeddings_dict = _merge_synonmys(food_to_embeddings_dict, max_sentence_count)

    with food_to_embeddings_dict_path.open('wb') as f:
        pickle.dump(food_to_embeddings_dict, f)

    return food_to_embeddings_dict
