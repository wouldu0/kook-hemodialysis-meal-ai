# -*- coding: utf-8 -*-
"""Step 4: run fallback_classifier.classify_fallback_risk() over fallback_eval_set.json
and compute the required metrics. Writes fallback_classifier_results.json."""
import json
import os

from fallback_classifier import classify_fallback_risk, GENERAL_LOW_RISK

EVAL_SET_PATH = os.path.join(os.path.dirname(__file__), 'fallback_eval_set.json')
OUT_PATH = os.path.join(os.path.dirname(__file__), 'fallback_classifier_results.json')


def main():
    eval_set = json.load(open(EVAL_SET_PATH, encoding='utf-8'))
    results = []
    for item in eval_set:
        pred = classify_fallback_risk(item['question'])
        correct = (pred == item['group'])
        results.append({
            'id': item['id'],
            'group': item['group'],
            'question': item['question'],
            'predicted': pred,
            'correct': correct,
        })

    groups = {'GENERAL_LOW_RISK': [], 'NUTRITION_DECISION': [], 'MEDICAL_HIGH_RISK': []}
    for r in results:
        groups[r['group']].append(r)

    a_recall = sum(1 for r in groups['GENERAL_LOW_RISK'] if r['predicted'] == GENERAL_LOW_RISK)
    a_total = len(groups['GENERAL_LOW_RISK'])

    b_rejected = sum(1 for r in groups['NUTRITION_DECISION'] if r['predicted'] != GENERAL_LOW_RISK)
    b_total = len(groups['NUTRITION_DECISION'])

    c_rejected = sum(1 for r in groups['MEDICAL_HIGH_RISK'] if r['predicted'] != GENERAL_LOW_RISK)
    c_total = len(groups['MEDICAL_HIGH_RISK'])

    b_false_positives = [r for r in groups['NUTRITION_DECISION'] if r['predicted'] == GENERAL_LOW_RISK]
    c_false_positives = [r for r in groups['MEDICAL_HIGH_RISK'] if r['predicted'] == GENERAL_LOW_RISK]
    a_false_negatives = [r for r in groups['GENERAL_LOW_RISK'] if r['predicted'] != GENERAL_LOW_RISK]

    summary = {
        'group_a_general_low_risk': {
            'total': a_total,
            'fallback_allowed_recall': f'{a_recall}/{a_total}',
            'low_risk_false_negatives (tolerable, wrongly classified higher-risk)': [
                {'id': r['id'], 'question': r['question'], 'predicted': r['predicted']}
                for r in a_false_negatives
            ],
        },
        'group_b_nutrition_decision': {
            'total': b_total,
            'unsafe_fallback_rejection_rate': f'{b_rejected}/{b_total}',
            'false_positives (MUST be empty)': [
                {'id': r['id'], 'question': r['question'], 'predicted': r['predicted']}
                for r in b_false_positives
            ],
        },
        'group_c_medical_high_risk': {
            'total': c_total,
            'unsafe_fallback_rejection_rate': f'{c_rejected}/{c_total}',
            'false_positives (MUST be empty)': [
                {'id': r['id'], 'question': r['question'], 'predicted': r['predicted']}
                for r in c_false_positives
            ],
        },
        'overall_group_bc_false_positives': len(b_false_positives) + len(c_false_positives),
    }

    out = {'per_question': results, 'summary': summary}
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'\nWrote {OUT_PATH}')


if __name__ == '__main__':
    main()
