# -*- coding: utf-8 -*-
"""
reranking_fetch_pool.py — one-shot LIVE retrieval to build the raw Top-10 candidate pool
used by reranking_experiment.py. Calls FOOK_rag_chatbot.retrieve(question, top_k=10) EXACTLY
ONCE per question (30 positive 'rag' questions + 21 'out_of_scope' questions), and dumps the
full (untruncated) text/source/score for every candidate to a JSON file. Every reranking
method compared in reranking_experiment.py re-uses this single dump — no re-embedding per
method, per the task's hard constraint.

Does NOT modify FOOK_rag_chatbot.py / FOOK_build_rag_kb.py / any KB file — read-only import.

Run: cd backend && source .env-loaded-env && venv/bin/python3 evaluation/reranking_fetch_pool.py
"""
import os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/


def main():
    if not os.environ.get('OPENAI_API_KEY'):
        print('[reranking_fetch_pool] OPENAI_API_KEY not set — aborting (no live embedding calls possible).')
        return 1

    import FOOK_rag_chatbot as C

    qpath = os.path.join(os.path.dirname(__file__), 'rag_eval_questions_v2.json')
    questions = json.load(open(qpath, encoding='utf-8'))
    rag_qs = [q for q in questions if q['type'] == 'rag']
    oos_qs = [q for q in questions if q['type'] == 'out_of_scope']
    print(f'rag questions: {len(rag_qs)}, out_of_scope questions: {len(oos_qs)}')

    out = {'RAG_MIN_SCORE': C.RAG_MIN_SCORE, 'rag': [], 'out_of_scope': []}

    for q in rag_qs:
        t0 = time.time()
        hits = C.retrieve(q['question'], top_k=10)
        dt = time.time() - t0
        out['rag'].append({
            'id': q['id'], 'category': q['category'], 'question': q['question'],
            'expected_source': q['expected_source'], 'note': q.get('note', ''),
            'pool': [{'text': h['text'], 'source': h['source'], 'score': h['score']} for h in hits],
            'elapsed_s': round(dt, 3),
        })
        print(f"  [rag] {q['id']:12s} top1={hits[0]['score']:.4f} src={hits[0]['source']}  ({dt:.2f}s)")

    for q in oos_qs:
        t0 = time.time()
        hits = C.retrieve(q['question'], top_k=10)
        dt = time.time() - t0
        out['out_of_scope'].append({
            'id': q['id'], 'category': q['category'], 'question': q['question'],
            'pool': [{'text': h['text'], 'source': h['source'], 'score': h['score']} for h in hits],
            'elapsed_s': round(dt, 3),
        })
        print(f"  [oos] {q['id']:12s} top1={hits[0]['score']:.4f} src={hits[0]['source']}  ({dt:.2f}s)")

    outpath = os.path.join(os.path.dirname(__file__), 'reranking_top10_pool.json')
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\nwrote {outpath}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
