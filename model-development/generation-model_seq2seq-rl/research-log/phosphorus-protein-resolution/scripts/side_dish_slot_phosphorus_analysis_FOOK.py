# -*- coding: utf-8 -*-
"""
side_dish_slot_phosphorus_analysis_FOOK.py — side_dish_slot_phosphorus_trace.csv(3,360행,
모델실행 결과)를 pandas로 재분석: 인 초과 유형 분류(A~G) + counterfactual 슬롯 시나리오 +
핵심 부찬 5종 심층분석. 모델/레버 코드 재실행 없음(순수 CSV 분석).

분류 기준(우선순위 기반, 서로 배타적 — 상세는 보고서 §4 참고):
  실패후보(after_pass=False) 중
  F(before_pass True→after False)를 최우선 분리 → 나머지(G 상위그룹) 안에서
  A(주찬지배) > B(부찬단독) > C(국단독) > D(주찬+부찬조합) > E(다슬롯누적) 순으로 첫 매치.
  G = {A,B,C,D,E}의 합집합(사용자 정의상 "adjust 전후 모두 실패"의 상위개념).
"""
import os
import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(CODE, 'side_dish_slot_phosphorus_diagnosis_out')
OUT_DIR = IN_DIR
TRACE_CSV = os.path.join(IN_DIR, 'side_dish_slot_phosphorus_trace.csv')

TARGET5 = ['숙주나물', '취나물무침', '애호박나물', '쑥갓두부무침', '새송이버섯볶음']


def main():
    df = pd.read_csv(TRACE_CSV, encoding='utf-8-sig')
    n_total = len(df)
    PMAX = df['phosphorus_limit'].iloc[0]
    print(f'로드: {TRACE_CSV} ({n_total}행), Pmax={PMAX:.2f}mg')

    df['before_pass'] = df['phosphorus_total_before'] < PMAX
    df['after_pass'] = df['phosphorus_total_after'] < PMAX
    df['phosphorus_high_fail'] = ~df['after_pass']

    # ────────────────────────────────────────────────────────────
    # 인 초과 유형 분류 (A~G, 우선순위 기반 배타적)
    # ────────────────────────────────────────────────────────────
    def classify(r):
        if r['after_pass']:
            return 'pass'
        if r['before_pass']:
            return 'F_레버후신규실패'
        rice, soup, main, side, kimchi = (r['rice_phosphorus_after'], r['soup_phosphorus_after'],
                                           r['main_phosphorus_after'], r['side_phosphorus_after'],
                                           r['kimchi_phosphorus_after'])
        total = r['phosphorus_total_after']
        main_share = r['main_phosphorus_share']
        remain_after_main = PMAX - (rice + soup + main)
        if main_share >= 0.5 or remain_after_main <= 0.15 * PMAX:
            return 'A_주찬단독지배형'
        if (total - side) < PMAX:
            return 'B_부찬직접초과형'
        if (total - soup) < PMAX:
            return 'C_국직접초과형'
        if (total - main - side) < PMAX:
            return 'D_주찬부찬조합형'
        return 'E_다슬롯누적형'

    df['phosphorus_failure_type'] = df.apply(classify, axis=1)

    detail_cols = ['anchor_type', 'seed_id', 'call_id', 'candidate_id', 'rice', 'soup', 'main_dish',
                    'side_dish', 'kimchi', 'phosphorus_total_before', 'phosphorus_total_after',
                    'rice_phosphorus_before', 'soup_phosphorus_before', 'main_phosphorus_before',
                    'side_phosphorus_before', 'kimchi_phosphorus_before',
                    'rice_phosphorus_after', 'soup_phosphorus_after', 'main_phosphorus_after',
                    'side_phosphorus_after', 'kimchi_phosphorus_after',
                    'phosphorus_excess_after', 'nutrition_all_pass', 'phosphorus_failure_type']
    df[detail_cols].to_csv(os.path.join(OUT_DIR, 'side_dish_slot_phosphorus_trace_classified.csv'),
                            index=False, encoding='utf-8-sig')

    # ────────────────────────────────────────────────────────────
    # 1) side_dish_slot_phosphorus_summary.csv
    # ────────────────────────────────────────────────────────────
    SLOTS = ['rice', 'soup', 'main', 'side', 'kimchi']
    summary_rows = []
    for scope_name, sub in [('overall', df)] + [('anchor', df[df['anchor_type'] == a]) for a in df['anchor_type'].unique()]:
        anchor_label = 'ALL' if scope_name == 'overall' else sub['anchor_type'].iloc[0]
        high = sub[sub['phosphorus_high_fail']]
        passed = sub[~sub['phosphorus_high_fail']]
        for slot in SLOTS:
            col = f'{slot}_phosphorus_after'
            share_col = f'{slot}_phosphorus_share'
            # counterfactual: 이 슬롯의 인 기여를 전부 제거하면 몇 명이 구제되는가 (실패후보 대상)
            if len(high):
                cf_total = high['phosphorus_total_after'] - high[col]
                rescued = int((cf_total < PMAX).sum())
            else:
                rescued = 0
            summary_rows.append({
                'scope': scope_name, 'anchor_type': anchor_label, 'slot': slot,
                'mean_phosphorus': sub[col].mean(), 'median_phosphorus': sub[col].median(),
                'mean_share': sub[share_col].mean(),
                'phosphorus_high_candidate_mean': high[col].mean() if len(high) else np.nan,
                'phosphorus_pass_candidate_mean': passed[col].mean() if len(passed) else np.nan,
                'counterfactual_rescued_count': rescued,
                'counterfactual_rescued_rate_of_fails': rescued / len(high) if len(high) else np.nan,
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, 'side_dish_slot_phosphorus_summary.csv'), index=False, encoding='utf-8-sig')
    print(f'① side_dish_slot_phosphorus_summary.csv ({len(summary_df)}행)')

    # ────────────────────────────────────────────────────────────
    # 2) side_dish_phosphorus_failure_types.csv
    # ────────────────────────────────────────────────────────────
    fail_pop = df[df['phosphorus_high_fail']]
    type_rows = []
    for scope_name, sub in [('ALL', fail_pop)] + [(a, fail_pop[fail_pop['anchor_type'] == a]) for a in df['anchor_type'].unique()]:
        n_scope = len(sub)
        if n_scope == 0:
            continue
        for ftype, g in sub.groupby('phosphorus_failure_type'):
            type_rows.append({
                'anchor_type': scope_name, 'failure_type': ftype, 'candidate_count': len(g),
                'ratio': len(g) / n_scope, 'mean_excess': g['phosphorus_excess_after'].mean(),
                'top_main_dishes': '; '.join(f'{m}({c})' for m, c in g['main_dish'].value_counts().head(5).items()),
                'top_side_dishes': '; '.join(f'{m}({c})' for m, c in g['side_dish'].value_counts().head(5).items()),
                'top_soups': '; '.join(f'{m}({c})' for m, c in g['soup'].value_counts().head(5).items()),
            })
    type_df = pd.DataFrame(type_rows)
    type_df.to_csv(os.path.join(OUT_DIR, 'side_dish_phosphorus_failure_types.csv'), index=False, encoding='utf-8-sig')
    print(f'② side_dish_phosphorus_failure_types.csv ({len(type_df)}행)')

    # ────────────────────────────────────────────────────────────
    # 3) Counterfactual 슬롯 시나리오 (phosphorus_high 후보만 대상)
    # ────────────────────────────────────────────────────────────
    cf_rows = []

    def eval_scenario(scenario_name, sub_high, new_total_series):
        rescued_mask = new_total_series < PMAX
        rescued_count = int(rescued_mask.sum())
        full_pass_count = int((rescued_mask & sub_high['other4_pass_before_phos_removed']).sum())
        uniq_gain = sub_high.loc[rescued_mask, 'side_dish'].nunique()
        return {
            'scenario': scenario_name, 'candidate_count': len(sub_high), 'rescued_count': rescued_count,
            'rescued_rate': rescued_count / len(sub_high) if len(sub_high) else np.nan,
            'nutrition_full_pass_count': full_pass_count,
            'unique_side_gain': uniq_gain,
            'protein_regression_count': 0, 'calorie_regression_count': 0,
        }

    for scope_name, sub_high in [('ALL', fail_pop)] + [(a, fail_pop[fail_pop['anchor_type'] == a]) for a in df['anchor_type'].unique()]:
        if len(sub_high) == 0:
            continue
        total = sub_high['phosphorus_total_after']
        main = sub_high['main_phosphorus_after']; side = sub_high['side_phosphorus_after']
        soup = sub_high['soup_phosphorus_after']

        pool = df[df['anchor_type'] == scope_name] if scope_name != 'ALL' else df
        median_side_p = pool['side_phosphorus_after'].median()
        median_soup_p = pool['soup_phosphorus_after'].median()

        scenarios = [
            ('1_부찬제거', total - side),
            ('2_국제거', total - soup),
            ('3_주찬10%감소', total - 0.10 * main),
            ('4_부찬10%감소', total - 0.10 * side),
            ('5_주찬5%+부찬5%감소', total - 0.05 * main - 0.05 * side),
            ('6_부찬저인중앙값교체', total - side + median_side_p),
            ('7_국저인중앙값교체', total - soup + median_soup_p),
        ]
        for name, new_total in scenarios:
            row = eval_scenario(name, sub_high, new_total)
            row['anchor_type'] = scope_name
            if name.startswith('6') or name.startswith('7'):
                row['protein_regression_count'] = None
                row['calorie_regression_count'] = None
            cf_rows.append(row)

    cf_df = pd.DataFrame(cf_rows)[['anchor_type', 'scenario', 'candidate_count', 'rescued_count', 'rescued_rate',
                                    'nutrition_full_pass_count', 'unique_side_gain',
                                    'protein_regression_count', 'calorie_regression_count']]
    cf_df.to_csv(os.path.join(OUT_DIR, 'side_dish_phosphorus_counterfactual.csv'), index=False, encoding='utf-8-sig')
    print(f'③ side_dish_phosphorus_counterfactual.csv ({len(cf_df)}행)')

    # ────────────────────────────────────────────────────────────
    # 4) 핵심 부찬 5종 심층분석 (콘솔+파일 텍스트로 별도 저장, 보고서에 반영)
    # ────────────────────────────────────────────────────────────
    lines = []

    def w(*a):
        lines.append(' '.join(str(x) for x in a))

    for m in TARGET5:
        g = df[df['side_dish'] == m]
        if len(g) == 0:
            w(f'--- {m}: 이번 소규모 샘플(3,360건)에는 등장 없음 ---')
            continue
        w(f'--- {m} (n={len(g)}) ---')
        w(f'  평균 전체인={g["phosphorus_total_after"].mean():.1f}  평균주찬인={g["main_phosphorus_after"].mean():.1f}'
          f'  평균부찬인={g["side_phosphorus_after"].mean():.1f}  평균국인={g["soup_phosphorus_after"].mean():.1f}')
        w(f'  기여율: 주찬={g["main_phosphorus_share"].mean()*100:.1f}%  부찬={g["side_phosphorus_share"].mean()*100:.1f}%'
          f'  국={g["soup_phosphorus_share"].mean()*100:.1f}%  밥={g["rice_phosphorus_share"].mean()*100:.1f}%'
          f'  김치={g["kimchi_phosphorus_share"].mean()*100:.1f}%')
        w(f'  인초과후보비율={g["phosphorus_high_fail"].mean()*100:.1f}%')
        wo_side = g['phosphorus_total_after'] - g['side_phosphorus_after']
        w(f'  부찬제외시통과율={float((wo_side < PMAX).mean())*100:.1f}%  (실제통과율={float((~g["phosphorus_high_fail"]).mean())*100:.1f}%)')
        by_anchor = g.groupby('anchor_type').agg(
            n=('candidate_id', 'count'), pass_rate=('phosphorus_high_fail', lambda s: 1 - s.mean()),
            mean_main=('main_phosphorus_after', 'mean'), mean_side=('side_phosphorus_after', 'mean'))
        w('  앵커별:')
        w(by_anchor.to_string())
        w('')

    with open(os.path.join(OUT_DIR, 'target5_slot_detail.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('④ target5_slot_detail.txt')

    # 같은 부찬이 두부콩류에서만 실패, 다른 앵커에서 통과하는 후보 단위 예시
    example_rows = []
    for m in TARGET5:
        g = df[df['side_dish'] == m]
        tofu_fail = g[(g['anchor_type'] == '두부콩류') & (g['phosphorus_high_fail'])]
        other_pass = g[(g['anchor_type'] != '두부콩류') & (~g['phosphorus_high_fail'])]
        if len(tofu_fail) and len(other_pass):
            for _, r in tofu_fail.head(3).iterrows():
                example_rows.append({'side_dish': m, 'case': 'tofu_fail', 'anchor_type': r['anchor_type'],
                                      'candidate_id': r['candidate_id'], 'main_dish': r['main_dish'],
                                      'phosphorus_total_after': r['phosphorus_total_after'],
                                      'main_phosphorus_after': r['main_phosphorus_after'],
                                      'side_phosphorus_after': r['side_phosphorus_after']})
            for _, r in other_pass.head(3).iterrows():
                example_rows.append({'side_dish': m, 'case': 'other_anchor_pass', 'anchor_type': r['anchor_type'],
                                      'candidate_id': r['candidate_id'], 'main_dish': r['main_dish'],
                                      'phosphorus_total_after': r['phosphorus_total_after'],
                                      'main_phosphorus_after': r['main_phosphorus_after'],
                                      'side_phosphorus_after': r['side_phosphorus_after']})
    ex_df = pd.DataFrame(example_rows)
    ex_df.to_csv(os.path.join(OUT_DIR, 'target5_tofu_vs_other_examples.csv'), index=False, encoding='utf-8-sig')
    print(f'⑤ target5_tofu_vs_other_examples.csv ({len(ex_df)}행)')

    return df, summary_df, type_df, cf_df


if __name__ == '__main__':
    main()
