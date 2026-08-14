# -*- coding: utf-8 -*-
"""Step 6: generate simulated "general ChatGPT-style" fallback answers for the 15
Group A (GENERAL_LOW_RISK) questions. Deliberately does NOT use FOOK_rag_chatbot.py's
SYSTEM prompt or any KOOK-specific framing / KB context — the point is to simulate what
a generic LLM answer (no grounding, no app-specific framing) would look like, so we can
manually score how risky/appropriate that would be if shown to a hemodialysis patient
with a distinguishing label. Live OpenAI call (temperature low), evaluation-only.
Writes fallback_group_a_samples.json.
"""
import json
import os

from openai import OpenAI

EVAL_SET_PATH = os.path.join(os.path.dirname(__file__), 'fallback_eval_set.json')
OUT_PATH = os.path.join(os.path.dirname(__file__), 'fallback_group_a_samples.json')

GENERIC_SYSTEM = (
    "You are a helpful general-knowledge assistant. Answer the user's question in "
    "Korean, in 2-4 sentences, at a general/educational level. Do not give specific "
    "numeric dosing, allowance amounts, or medical/treatment advice. This is a plain "
    "general-knowledge answer, not a clinical consultation."
)

LABEL_CANDIDATES = [
    "[일반 AI 안내] 아래 내용은 KOOK의 근거 자료에서 확인한 답변이 아니라 일반적인 정보입니다.",
    "[일반 정보] KOOK 자료로 확인된 답변이 아닌, 일반 지식 기반 설명입니다.",
    "[AI 참고용] 이 답변은 KOOK 근거 자료가 아닌 일반 지식으로 작성되었습니다.",
]
RECOMMENDED_LABEL_INDEX = 0  # filled in narratively in the final report


def generate_one(client, question):
    resp = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": GENERIC_SYSTEM},
            {"role": "user", "content": question},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def main():
    eval_set = json.load(open(EVAL_SET_PATH, encoding='utf-8'))
    group_a = [q for q in eval_set if q['group'] == 'GENERAL_LOW_RISK']
    assert len(group_a) == 15

    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    results = []
    for item in group_a:
        raw_answer = generate_one(client, item['question'])
        labeled_variants = {
            f'label_candidate_{i+1}': f"{label}\n\n{raw_answer}"
            for i, label in enumerate(LABEL_CANDIDATES)
        }
        results.append({
            'id': item['id'],
            'question': item['question'],
            'raw_simulated_answer': raw_answer,
            'label_candidates': LABEL_CANDIDATES,
            'labeled_variants': labeled_variants,
            'recommended_label': LABEL_CANDIDATES[RECOMMENDED_LABEL_INDEX],
            'recommended_labeled_answer': labeled_variants[f'label_candidate_{RECOMMENDED_LABEL_INDEX+1}'],
        })
        print(f"--- {item['id']}: {item['question']}")
        print(raw_answer)
        print()

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == '__main__':
    main()
