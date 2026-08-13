# -*- coding: utf-8 -*-
"""
evaluation/evaluate_rag.py — RAG 챗봇(FOOK_rag_chatbot.py) 검색 품질 평가.

evaluation/rag_eval_questions.json의 질문들로:
  1) retrieve()의 Top-1/Top-3/Top-5 적중률 (RAG형 질문의 expected_source가 실제 검색 결과에
     그 순위 안에 들어오는가) — retrieve()는 원 점수 그대로 반환하는 함수이므로 게이트(RAG_MIN_SCORE)
     적용 전 "검색 자체"의 품질을 잰다.
  2) no-answer 정확도 — out-of-scope 질문에서 _relevant(retrieve(q)) (RAG_MIN_SCORE 게이트 통과분)가
     실제로 비어 있는가(=서비스가 "확인할 수 없다"고 올바르게 거절할지).
  3) 라우팅 정확도 — find_food()가 food_db형 질문은 올바른 재료로, rag/out_of_scope형 질문은
     None으로(엉뚱하게 재료질문으로 오인하지 않고) 분기하는가.

임베딩 API(retrieve() 내부의 OpenAI embeddings 호출)만 실제로 호출한다 — 챗 콤플리션은
호출하지 않는다(위 세 지표 모두 임베딩 검색·순수 함수만으로 계산 가능해서 불필요한 비용/토큰을
쓰지 않는다). OPENAI_API_KEY가 없으면 라이브 임베딩 호출 없이 "스킵" 메시지만 내고 0으로 종료한다
(실패로 죽지 않음 — CI/평소 개발 환경에 키가 없는 게 정상이므로).

실행: cd backend && python evaluation/evaluate_rag.py [--out evaluation/rag_eval_results.json]
"""
from __future__ import annotations
import os, sys, json, argparse, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ 를 import 경로에 추가


def load_questions():
    path = os.path.join(os.path.dirname(__file__), 'rag_eval_questions.json')
    return json.load(open(path, encoding='utf-8'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'rag_eval_results.json'))
    ap.add_argument('--top_k', type=int, default=5)
    args = ap.parse_args()

    if not os.environ.get('OPENAI_API_KEY'):
        print('[evaluate_rag] OPENAI_API_KEY가 설정되어 있지 않습니다 — 실제 임베딩 호출이 필요한 '
              '평가라 이 스크립트는 건너뜁니다(오류 아님). '
              'OPENAI_API_KEY를 설정한 뒤 다시 실행하세요.')
        return 0

    import FOOK_rag_chatbot as C

    questions = load_questions()
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
            # rag/out_of_scope 결과 항목에 라우팅 정오만 덧붙인다 (이미 위에서 추가된 항목을 찾아서 갱신)
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
    }

    out = {'summary': summary, 'results': results}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'\n상세 결과: {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
