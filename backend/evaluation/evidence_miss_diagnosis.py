# -*- coding: utf-8 -*-
"""
evidence_miss_diagnosis.py — Recall@10 evidence-miss root-cause diagnosis (2026-08-14).

Confirmed Top-10 evidence misses (from manual re-verification of topk_experiment_raw.json
against rag_evidence_inspection_v2.md and direct KB source-file greps — see report):
  p2, prot2, snack1, fish1, transplant1, itch1

For each Coverage-YES miss, this script calls the PRODUCTION retrieve() function live at
top_k=30 to find the actual rank/score of the chunk containing the real KB evidence
(matched by source + a distinctive substring found via direct grep of data/rag-source/*.txt).

It also runs one controlled query-rewrite experiment per candidate (itch1, p2) comparing
original vs. rewritten phrasing.

Read-only against production code: only imports retrieve() from FOOK_rag_chatbot.py.
Does not modify KB, chunking, embeddings, or RAG_MIN_SCORE.
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from FOOK_rag_chatbot import retrieve  # noqa: E402

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evidence_miss_diagnosis_results.json')

# (qid, question, evidence_source, distinctive_substring_to_match)
MISS_QUESTIONS = [
    ("p2", "가공식품에 들어있는 '숨은 인'이 왜 문제가 되나요?",
     "혈액투석_영양식생활관리_2권", "숨은 인"),
    ("prot2", "투석을 받고 있는 환자는 단백질을 얼마나 섭취해야 하나요?",
     "콩팥병_환자를_위한_안내서", "1.0-1.2 g/kg/day로 늘려야"),
    ("snack1", "혈액투석 환자가 먹을 수 있는 간식 레시피를 알려주세요.",
     "혈액투석환자를_위한_간식", "슈가토스트"),
    ("fish1", "생선은 얼마나 드셔도 되나요?",
     "혈액투석_영양식생활관리_2권", "고기, 생선, 달걀, 콩류"),
    ("transplant1", "신장이식을 받은 직후에는 물을 얼마나 마셔야 하나요?",
     "콩팥병_환자를_위한_안내서", "3 L 이상의 물을 필요로 할 수 있습니다"),
    ("itch1", "투석하고 나서부터 몸이 자꾸 가려운데 왜 그런 거예요?",
     "혈액투석_영양식생활관리_2권", "요독성성 가려움증의 치료"),
]

# one honest query rewrite per candidate we deep-dive on wording mismatch
REWRITES = {
    "itch1": "혈액투석 환자의 소양증(가려움증) 원인",
    "p2": "가공식품 첨가물 인산염 흡수율",
}

TOP_K = 30


def find_rank(results, source, substring):
    for i, r in enumerate(results, start=1):
        if r['source'] == source and substring.replace(' ', '') in r['text'].replace(' ', ''):
            return i, r['score']
    return None, None


def main():
    out = {}
    for qid, question, ev_source, ev_substring in MISS_QUESTIONS:
        print(f"=== {qid}: {question} ===", flush=True)
        results = retrieve(question, top_k=TOP_K)
        rank, score = find_rank(results, ev_source, ev_substring)
        top1 = results[0]
        entry = {
            "question": question,
            "evidence_source": ev_source,
            "evidence_substring": ev_substring,
            "top1_score": top1['score'],
            "top1_source": top1['source'],
            "top1_text_head": top1['text'][:120],
            "evidence_rank_in_top30": rank,
            "evidence_score": score,
            "score_gap_vs_top1": (top1['score'] - score) if score is not None else None,
            "top10_sources": [r['source'] for r in results[:10]],
            "top10_scores": [round(r['score'], 4) for r in results[:10]],
        }
        print(f"  top1: {top1['source']} score={top1['score']:.4f}")
        print(f"  evidence found at rank={rank} score={score}")
        out[qid] = entry

        if qid in REWRITES:
            rw_q = REWRITES[qid]
            rw_results = retrieve(rw_q, top_k=TOP_K)
            rw_rank, rw_score = find_rank(rw_results, ev_source, ev_substring)
            print(f"  REWRITE '{rw_q}' -> evidence rank={rw_rank} score={rw_score}")
            entry['rewrite'] = {
                "rewritten_query": rw_q,
                "evidence_rank_in_top30": rw_rank,
                "evidence_score": rw_score,
                "top1_score": rw_results[0]['score'],
                "top1_source": rw_results[0]['source'],
                "top1_text_head": rw_results[0]['text'][:120],
            }

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == '__main__':
    main()
