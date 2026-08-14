# -*- coding: utf-8 -*-
"""Step 2 audit script (evaluation-only, not imported by production code).
Runs find_food() -> classify_scope() -> retrieve()/_relevant() over the fallback eval
set using the LIVE production functions in FOOK_rag_chatbot.py (no monkeypatching,
no reimplementation) and records the resulting current-behavior category for each
question. Does NOT call the chat-completion LLM (not needed for this audit — retrieve()
already needs a live embeddings call, which is the only live API usage here).
Writes backend/evaluation/fallback_current_behavior.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FOOK_rag_chatbot as C

EVAL_SET_PATH = os.path.join(os.path.dirname(__file__), 'fallback_eval_set.json')
OUT_PATH = os.path.join(os.path.dirname(__file__), 'fallback_current_behavior.json')


def audit_one(question):
    rec = {'question': question}

    food = C.find_food(question)
    rec['find_food_match'] = food

    if food:
        rec['classify_scope_verdict'] = None
        rec['classify_scope_reason'] = None
        rec['retrieve_top1_score'] = None
        rec['passes_rag_min_score'] = None
        rec['final_category'] = 'food_db'
        return rec

    scope = C.classify_scope(question)
    rec['classify_scope_verdict'] = scope.verdict
    rec['classify_scope_reason'] = scope.reason

    if scope.verdict == C.OUT_OF_SCOPE:
        rec['retrieve_top1_score'] = None
        rec['passes_rag_min_score'] = None
        rec['final_category'] = 'out_of_scope'
        return rec

    # IN_SCOPE -> live retrieve() (needs OPENAI_API_KEY for embeddings)
    hits = C.retrieve(question, top_k=C.RAG_CANDIDATE_POOL)
    top1 = hits[0]['score'] if hits else None
    rec['retrieve_top1_score'] = top1
    relevant = C._relevant(hits)
    rec['passes_rag_min_score'] = bool(relevant)
    rec['final_category'] = 'grounded_rag' if relevant else 'no_evidence'
    return rec


def main():
    eval_set = json.load(open(EVAL_SET_PATH, encoding='utf-8'))
    results = []
    for item in eval_set:
        rec = audit_one(item['question'])
        rec['id'] = item['id']
        rec['group'] = item['group']
        results.append(rec)
        print(f"{item['id']:5s} [{item['group']:18s}] -> {rec['final_category']:12s} "
              f"(find_food={rec['find_food_match']}, scope={rec['classify_scope_reason']}, "
              f"top1={rec['retrieve_top1_score']})")

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {OUT_PATH}")

    # quick summary
    from collections import Counter
    cat_counts = Counter(r['final_category'] for r in results)
    print("Category counts:", dict(cat_counts))
    by_group = {}
    for r in results:
        by_group.setdefault(r['group'], Counter())[r['final_category']] += 1
    for g, c in by_group.items():
        print(g, dict(c))


if __name__ == '__main__':
    main()
