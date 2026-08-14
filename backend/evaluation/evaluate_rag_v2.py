# -*- coding: utf-8 -*-
"""
evaluation/evaluate_rag_v2.py — evaluate_rag.py의 v2 버전. 원본 evaluate_rag.py는 전혀 수정하지
않았다(그대로 두면 누구든 원본 baseline을 그대로 재현할 수 있다). 이 파일은 그 사본에 세 가지만
더한 것이다:

  1) --questions 인자로 다른 질문 세트 파일을 받을 수 있음 (기본값 rag_eval_questions_v2.json).
     expected_source는 원본 스크립트와 마찬가지로 리스트로 두고 set 교집합으로 hit을 판정한다
     (원본 evaluate_rag.py가 이미 `expected = set(q['expected_source'])`로 다중 출처를 지원하고
     있었으므로 — 별도 스키마 변경이 필요 없었다. 즉 여러 출처 중 하나만 맞아도 hit).
  2) rag형 질문 결과에 top_texts(각 hit의 청크 앞부분 400자)를 같이 저장한다 — evidence-level
     inspection(사람이 실제로 검색된 청크 내용을 읽고 "이 근거로 실제 답변 가능한가"를 판단하는 절차)에
     원 텍스트가 필요하기 때문. 원본 스크립트는 소스명/점수만 남겨 사람이 원문을 다시 볼 수 없었다.
  3) out_of_scope 결과에도 동일하게 top_texts를 저장한다(오탐 원인 진단용).

실행: cd backend && python evaluation/evaluate_rag_v2.py
  [--questions evaluation/rag_eval_questions_v2.json] [--out evaluation/rag_eval_results_v2.json]
"""
from __future__ import annotations
import os, sys, json, argparse, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ 를 import 경로에 추가


def load_questions(path):
    return json.load(open(path, encoding='utf-8'))


def _snippet(text, n=400):
    t = text.strip().replace('\n', ' ')
    return t[:n] + ('…' if len(t) > n else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--questions', default=os.path.join(os.path.dirname(__file__), 'rag_eval_questions_v2.json'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'rag_eval_results_v2.json'))
    ap.add_argument('--top_k', type=int, default=5)
    args = ap.parse_args()

    if not os.environ.get('OPENAI_API_KEY'):
        print('[evaluate_rag_v2] OPENAI_API_KEY가 설정되어 있지 않습니다 — 실제 임베딩 호출이 필요한 '
              '평가라 이 스크립트는 건너뜁니다(오류 아님). '
              'OPENAI_API_KEY를 설정한 뒤 다시 실행하세요.')
        return 0

    import FOOK_rag_chatbot as C

    questions = load_questions(args.questions)
    results = []
    rag_qs = [q for q in questions if q['type'] == 'rag']
    oos_qs = [q for q in questions if q['type'] == 'out_of_scope']
    food_qs = [q for q in questions if q['type'] == 'food_db']

    # ── 1) Top-1/3/5 적중률 (RAG형) ──────────────────────────────────────────
    hit1 = hit3 = hit5 = 0
    for q in rag_qs:
        t0 = time.time()
        hits = C.retrieve(q['question'], top_k=args.top_k)
        dt = time.time() - t0
        got_sources = [h['source'] for h in hits]
        expected = set(q['expected_source'])
        h1 = bool(expected & set(got_sources[:1]))
        h3 = bool(expected & set(got_sources[:3]))
        h5 = bool(expected & set(got_sources[:5]))
        hit1 += h1; hit3 += h3; hit5 += h5
        relevant = C._relevant(hits)
        results.append({
            'id': q['id'], 'category': q['category'], 'type': 'rag', 'question': q['question'],
            'expected_source': q['expected_source'], 'top_sources': got_sources,
            'top_scores': [round(h['score'], 4) for h in hits],
            'top_texts': [_snippet(h['text']) for h in hits],
            'hit@1': h1, 'hit@3': h3, 'hit@5': h5,
            'n_relevant_after_gate': len(relevant), 'elapsed_s': round(dt, 3),
        })
    n_rag = max(1, len(rag_qs))

    # ── 2) no-answer 정확도 (out-of-scope형) ─────────────────────────────────
    correct_no_answer = 0
    for q in oos_qs:
        hits = C.retrieve(q['question'], top_k=args.top_k)
        relevant = C._relevant(hits)
        correct = len(relevant) == 0
        correct_no_answer += int(correct)
        results.append({
            'id': q['id'], 'category': q['category'], 'type': 'out_of_scope', 'question': q['question'],
            'top_sources': [h['source'] for h in hits],
            'top_scores': [round(h['score'], 4) for h in hits],
            'top_texts': [_snippet(h['text']) for h in hits],
            'correctly_refused': correct,
        })
    n_oos = max(1, len(oos_qs))

    # ── 3) 라우팅 정확도 (find_food) — API 호출 없음 ─────────────────────────
    routing_correct = 0
    n_routing = 0
    for q in questions:
        n_routing += 1
        food = C.find_food(q['question'])
        if q['type'] == 'food_db':
            ok = food is not None and food.startswith(q['expected_food_prefix'])
            routing_correct += int(ok)
            results.append({'id': q['id'], 'category': q['category'], 'type': 'food_db',
                             'question': q['question'], 'expected_food_prefix': q['expected_food_prefix'],
                             'matched_food': food, 'routing_correct': ok})
        else:
            ok = food is None
            routing_correct += int(ok)
            for r in results:
                if r['id'] == q['id']:
                    r['routing_correct'] = ok
                    break

    summary = {
        'n_questions': len(questions),
        'n_rag': len(rag_qs), 'n_out_of_scope': len(oos_qs), 'n_food_db': len(food_qs),
        'top1_hit_rate': round(hit1 / n_rag, 3),
        'top3_hit_rate': round(hit3 / n_rag, 3),
        'top5_hit_rate': round(hit5 / n_rag, 3),
        'no_answer_accuracy': round(correct_no_answer / n_oos, 3),
        'routing_accuracy': round(routing_correct / max(1, n_routing), 3),
        'RAG_MIN_SCORE': C.RAG_MIN_SCORE,
        'questions_file': args.questions,
    }

    out = {'summary': summary, 'results': results}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'\n상세 결과: {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
