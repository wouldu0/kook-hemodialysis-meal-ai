# -*- coding: utf-8 -*-
"""
track1_rl_final_analysis_FOOK.py — track1_rl_final_eval_FOOK.py가 저장한 raw_candidates.pkl을
읽어 전체 지표/전이분석/대표사례/CSV 6종 + 최종 보고서를 생성한다. (읽기 전용 분석, 재생성 없음)

4개 조합: A=BASE+OLD, B=BASE+NEW, C=RL+NEW(실제 production), D=RL+OLD
"""
import os, sys, json, pickle, csv
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy import stats

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

CODE = r'E:\final\Diet-Generation-As-Sequence-master\Diet-Generation-As-Sequence-master\Code'
OUT_DIR = os.path.join(CODE, 'track1_rl_final_eval_out')

with open(os.path.join(OUT_DIR, 'raw_candidates.pkl'), 'rb') as f:
    raw = pickle.load(f)
print(f'로드된 raw 후보: {len(raw)}건')

NUT_KEYS = ('E', 'protein', 'K', 'P', 'Na_season')

rows = []
for c in raw:
    inst_o, inst_n = c['inst_OLD'], c['inst_NEW']
    fm_o = list(dict.fromkeys(i['menu'] for i in inst_o))
    fm_n = list(dict.fromkeys(i['menu'] for i in inst_n))
    changed_o = any(abs(c['after_OLD'][k] - c['before_OLD'][k]) > 1e-6 for k in NUT_KEYS)
    changed_n = any(abs(c['after_NEW'][k] - c['before_NEW'][k]) > 1e-6 for k in NUT_KEYS)
    row = {
        'candidate_id': c['candidate_id'], 'anchor_type': c['anchor_type'], 'model': c['model'],
        'seed_id': c['seed_id'], 'call_id': c['call_id'], 'pos_in_batch': c['pos_in_batch'],
        'menus': '|'.join(c['menus']), 'is_plant_protein': c['is_plant_protein'],
        'valid_seq': c['valid_seq'], 'dup_slot': c['dup_slot'], 'anchor_present': c['anchor_present'],
        'before_E': c['before_NEW']['E'], 'before_protein': c['before_NEW']['protein'],
        'before_K': c['before_NEW']['K'], 'before_P': c['before_NEW']['P'], 'before_Na': c['before_NEW']['Na_season'],
        'before_P_matches_OLD': abs(c['before_OLD']['P'] - c['before_NEW']['P']) < 1e-6,
    }
    for lv, after_key, flag_key, unreal_key, el_key, fm, changed in [
        ('OLD', 'after_OLD', 'flags_OLD', 'unreal_OLD', 'elapsed_OLD', fm_o, changed_o),
        ('NEW', 'after_NEW', 'flags_NEW', 'unreal_NEW', 'elapsed_NEW', fm_n, changed_n)]:
        after = c[after_key]; flags = c[flag_key]
        row[f'{lv}_E'] = after['E']; row[f'{lv}_protein'] = after['protein']
        row[f'{lv}_K'] = after['K']; row[f'{lv}_P'] = after['P']; row[f'{lv}_Na'] = after['Na_season']
        row[f'{lv}_all_pass'] = flags['all_pass']; row[f'{lv}_raw_pass'] = flags['raw_pass']
        row[f'{lv}_protein_pass'] = flags['protein_pass']; row[f'{lv}_calorie_pass'] = flags['calorie_pass']
        row[f'{lv}_na_pass'] = flags['na_pass']; row[f'{lv}_k_pass'] = flags['k_pass']
        row[f'{lv}_protein_low'] = flags['protein_low']
        row[f'{lv}_unreal'] = c[unreal_key] is not None
        row[f'{lv}_elapsed_ms'] = c[el_key] * 1000
        row[f'{lv}_score'] = sum([flags['raw_pass'], flags['protein_pass'], flags['calorie_pass'], flags['na_pass'], flags['k_pass']])
        row[f'{lv}_final_menus'] = '|'.join(fm)
        row[f'{lv}_changed'] = changed
    rows.append(row)

df = pd.DataFrame(rows)
print(f'DataFrame 생성 완료: {df.shape}')

# 앵커별 지정 앵커명 (anchor_present/anchor_preserved 재계산에 사용)
ANCHOR_MENU = {'두부콩류': '두부양념조림', '생선구이': '고등어구이', '육류': '제육불고기', '랜덤': None}
for lv in ('OLD', 'NEW'):
    def _preserved(r, lv=lv):
        am = ANCHOR_MENU[r['anchor_type']]
        if am is None:
            return None
        return am in r[f'{lv}_final_menus'].split('|')
    df[f'{lv}_anchor_preserved_after'] = df.apply(_preserved, axis=1)

CONDITIONS = {'A': ('BASE', 'OLD'), 'B': ('BASE', 'NEW'), 'C': ('RL', 'NEW'), 'D': ('RL', 'OLD')}
ANCHORS = ['두부콩류', '생선구이', '육류', '랜덤']

# ============================================================
# 1. 배치 단위 선택 시뮬레이션 (make_meal 로직 재현: pos_in_batch 오름차순 첫 all_pass, 없으면 최고점수)
# ============================================================
def simulate_selection(group, lv):
    g = group.sort_values('pos_in_batch')
    passing = g[g[f'{lv}_all_pass']]
    if len(passing) > 0:
        sel = passing.iloc[0]
        rank = int(sel['pos_in_batch']) + 1
        has_pass = True
    else:
        idx = g[f'{lv}_score'].idxmax()
        sel = g.loc[idx]
        rank = None
        has_pass = False
    return sel, has_pass, rank


selection_rows = []
for cond, (model, lv) in CONDITIONS.items():
    sub = df[df['model'] == model]
    for (anchor_type, seed_id, call_id), g in sub.groupby(['anchor_type', 'seed_id', 'call_id']):
        sel, has_pass, rank = simulate_selection(g, lv)
        selection_rows.append({
            'condition': cond, 'model': model, 'lever': lv, 'anchor_type': anchor_type,
            'seed_id': seed_id, 'call_id': call_id, 'batch_success': has_pass, 'rank_of_first_pass': rank,
            'sel_all_pass': sel[f'{lv}_all_pass'], 'sel_protein_low': sel[f'{lv}_protein_low'],
            'sel_unreal': sel[f'{lv}_unreal'], 'sel_final_menus': sel[f'{lv}_final_menus'],
            'sel_menus_raw': sel['menus'], 'sel_score': sel[f'{lv}_score'],
            'sel_raw_pass': sel[f'{lv}_raw_pass'], 'sel_protein_pass': sel[f'{lv}_protein_pass'],
            'sel_calorie_pass': sel[f'{lv}_calorie_pass'], 'sel_na_pass': sel[f'{lv}_na_pass'],
            'sel_k_pass': sel[f'{lv}_k_pass'],
            'sel_anchor_preserved': sel.get(f'{lv}_anchor_preserved_after'),
        })
sel_df = pd.DataFrame(selection_rows)
print(f'배치 선택 시뮬레이션 완료: {sel_df.shape} (조건 {sel_df.condition.nunique()}개 x 배치)')


# ============================================================
# 2. 조건(A/B/C/D) x 앵커별 종합 지표
# ============================================================
def summarize(sub_df, sub_sel, cond_label, anchor_label):
    n_batches = len(sub_sel)
    n_cands = len(sub_df)
    lv = CONDITIONS[cond_label][1]
    out = {
        'condition': cond_label, 'anchor_type': anchor_label, 'n_batches': n_batches, 'n_candidates': n_cands,
        'batch_success_rate': sub_sel['batch_success'].mean() if n_batches else None,
        'zero_candidate_rate': 1 - sub_sel['batch_success'].mean() if n_batches else None,
        'final_5nutrient_pass_rate': sub_sel['sel_all_pass'].mean() if n_batches else None,
        'final_raw_P_pass_rate': sub_sel['sel_raw_pass'].mean() if n_batches else None,
        'final_protein_pass_rate': sub_sel['sel_protein_pass'].mean() if n_batches else None,
        'final_calorie_pass_rate': sub_sel['sel_calorie_pass'].mean() if n_batches else None,
        'final_na_pass_rate': sub_sel['sel_na_pass'].mean() if n_batches else None,
        'final_k_pass_rate': sub_sel['sel_k_pass'].mean() if n_batches else None,
        'final_protein_low_rate': sub_sel['sel_protein_low'].mean() if n_batches else None,
        'final_unrealistic_rate': sub_sel['sel_unreal'].mean() if n_batches else None,
        'fallback_usage_rate': 1 - sub_sel['batch_success'].mean() if n_batches else None,
        'mean_rank_of_first_pass': sub_sel['rank_of_first_pass'].dropna().mean() if sub_sel['rank_of_first_pass'].notna().any() else None,
        'mean_exec_time_ms': sub_df[f'{lv}_elapsed_ms'].mean() if n_cands else None,
        'lever_change_rate_all_candidates': sub_df[f'{lv}_changed'].mean() if n_cands else None,
        'anchor_preserved_rate': sub_sel['sel_anchor_preserved'].mean() if (anchor_label != '랜덤' and n_batches) else None,
        'unique_final_combo_count': sub_sel['sel_final_menus'].nunique() if n_batches else None,
        'unique_soup_count': sub_sel['sel_final_menus'].apply(lambda s: s.split('|')[1] if len(s.split('|')) > 1 else None).nunique() if n_batches else None,
        'unique_main_count': sub_sel['sel_final_menus'].apply(lambda s: s.split('|')[2] if len(s.split('|')) > 2 else None).nunique() if n_batches else None,
        'unique_side_count': sub_sel['sel_final_menus'].apply(lambda s: s.split('|')[3] if len(s.split('|')) > 3 else None).nunique() if n_batches else None,
        'unique_kimchi_count': sub_sel['sel_final_menus'].apply(lambda s: s.split('|')[-1]).nunique() if n_batches else None,
    }
    if n_batches:
        top = sub_sel['sel_final_menus'].value_counts(normalize=True)
        out['top1_menu_share'] = top.iloc[0] if len(top) else None
        out['top5_menu_share'] = top.iloc[:5].sum() if len(top) else None
    else:
        out['top1_menu_share'] = None; out['top5_menu_share'] = None
    return out


summary_rows = []
for cond, (model, lv) in CONDITIONS.items():
    sub_df_all = df[df['model'] == model]
    sub_sel_all = sel_df[sel_df['condition'] == cond]
    summary_rows.append(summarize(sub_df_all, sub_sel_all, cond, '전체'))
    for a in ANCHORS:
        sub_df_a = sub_df_all[sub_df_all['anchor_type'] == a]
        sub_sel_a = sub_sel_all[sub_sel_all['anchor_type'] == a]
        summary_rows.append(summarize(sub_df_a, sub_sel_a, cond, a))

summary_df = pd.DataFrame(summary_rows)
overall_df = summary_df[summary_df['anchor_type'] == '전체'].copy()
by_anchor_df = summary_df[summary_df['anchor_type'] != '전체'].copy()

overall_csv = os.path.join(OUT_DIR, 'track1_rl_final_comparison_summary.csv')
overall_df.to_csv(overall_csv, index=False, encoding='utf-8-sig')
by_anchor_csv = os.path.join(OUT_DIR, 'track1_metrics_by_anchor.csv')
by_anchor_df.to_csv(by_anchor_csv, index=False, encoding='utf-8-sig')
print(f'저장: {overall_csv}')
print(f'저장: {by_anchor_csv}')

print('\n=== 전체(4앵커 합계) 조건별 요약 ===')
for _, r in overall_df.iterrows():
    print(f"  [{r['condition']}={CONDITIONS[r['condition']]}] 성공률={r['batch_success_rate']*100:.1f}% "
          f"5영양통과={r['final_5nutrient_pass_rate']*100:.1f}% protein_low={r['final_protein_low_rate']*100:.1f}% "
          f"비현실={r['final_unrealistic_rate']*100:.1f}% 평균순위={r['mean_rank_of_first_pass']} "
          f"실행={r['mean_exec_time_ms']:.3f}ms 고유조합={r['unique_final_combo_count']} top1={r['top1_menu_share']*100:.1f}%")


# ============================================================
# 3. 레버효과 비교 (OLD vs NEW, 모델별 candidate-level paired) — track1_lever_effect_comparison.csv
# ============================================================
def identical_ab(r, model_lv_pair):
    return abs(r[f'{model_lv_pair[0]}_P'] - r[f'{model_lv_pair[1]}_P']) < 1e-9 and \
        abs(r[f'{model_lv_pair[0]}_protein'] - r[f'{model_lv_pair[1]}_protein']) < 1e-9 and \
        abs(r[f'{model_lv_pair[0]}_E'] - r[f'{model_lv_pair[1]}_E']) < 1e-9 and \
        abs(r[f'{model_lv_pair[0]}_Na'] - r[f'{model_lv_pair[1]}_Na']) < 1e-9 and \
        abs(r[f'{model_lv_pair[0]}_K'] - r[f'{model_lv_pair[1]}_K']) < 1e-9


lever_effect_rows = []
for model in ('BASE', 'RL'):
    sub = df[df['model'] == model]
    for a in ANCHORS + ['전체']:
        s = sub if a == '전체' else sub[sub['anchor_type'] == a]
        if len(s) == 0:
            continue
        ident = s.apply(lambda r: identical_ab(r, ('OLD', 'NEW')), axis=1)
        improved = (~s['OLD_all_pass']) & s['NEW_all_pass']
        worsened = s['OLD_all_pass'] & (~s['NEW_all_pass'])
        both_pass = s['OLD_all_pass'] & s['NEW_all_pass']
        both_fail = (~s['OLD_all_pass']) & (~s['NEW_all_pass'])
        lever_effect_rows.append({
            'model': model, 'anchor_type': a, 'n_candidates': len(s),
            'is_plant_protein_trigger_rate': s['is_plant_protein'].mean(),
            'identical_rate': ident.mean(), 'diff_rate': 1 - ident.mean(),
            'OLD_all_pass_rate': s['OLD_all_pass'].mean(), 'NEW_all_pass_rate': s['NEW_all_pass'].mean(),
            'improved_rate': improved.mean(), 'worsened_rate': worsened.mean(),
            'both_pass_rate': both_pass.mean(), 'both_fail_rate': both_fail.mean(),
            'OLD_unreal_rate': s['OLD_unreal'].mean(), 'NEW_unreal_rate': s['NEW_unreal'].mean(),
            'OLD_protein_low_rate': s['OLD_protein_low'].mean(), 'NEW_protein_low_rate': s['NEW_protein_low'].mean(),
            'OLD_changed_rate': s['OLD_changed'].mean(), 'NEW_changed_rate': s['NEW_changed'].mean(),
            'mean_P_OLD': s['OLD_P'].mean(), 'mean_P_NEW': s['NEW_P'].mean(),
        })
lever_effect_df = pd.DataFrame(lever_effect_rows)
lever_effect_csv = os.path.join(OUT_DIR, 'track1_lever_effect_comparison.csv')
lever_effect_df.to_csv(lever_effect_csv, index=False, encoding='utf-8-sig')
print(f'\n저장: {lever_effect_csv}')
print('\n=== 레버효과(OLD->NEW, candidate-level paired) 두부콩류 ===')
for _, r in lever_effect_df[lever_effect_df['anchor_type'] == '두부콩류'].iterrows():
    print(f"  [{r['model']}] 두부판정율={r['is_plant_protein_trigger_rate']*100:.1f}% 동일율={r['identical_rate']*100:.1f}% "
          f"개선={r['improved_rate']*100:.1f}% 악화={r['worsened_rate']*100:.1f}% "
          f"OLD통과={r['OLD_all_pass_rate']*100:.1f}%->NEW통과={r['NEW_all_pass_rate']*100:.1f}%")
print('\n=== 비대상 앵커(생선구이/육류) 동일율 확인 (100% 기대) ===')
for _, r in lever_effect_df[lever_effect_df['anchor_type'].isin(['생선구이', '육류'])].iterrows():
    print(f"  [{r['model']}/{r['anchor_type']}] 동일율={r['identical_rate']*100:.2f}% 두부판정율={r['is_plant_protein_trigger_rate']*100:.2f}%")


# ============================================================
# 4. Paired 전이분석 — track1_paired_transitions.csv
#    (a) 레버효과 전이: candidate-level, 모델별(BASE, RL)
#    (b) 모델효과 전이: batch-level, 레버버전별(OLD, NEW) — 동일 (anchor,seed,call) 배치의
#        선택결과(성공/실패)를 BASE vs RL로 비교
# ============================================================
transition_rows = []
for model in ('BASE', 'RL'):
    sub = df[df['model'] == model]
    for a in ANCHORS + ['전체']:
        s = sub if a == '전체' else sub[sub['anchor_type'] == a]
        if len(s) == 0:
            continue
        n = len(s)
        both_pass = (s['OLD_all_pass'] & s['NEW_all_pass']).sum()
        old_fail_new_pass = ((~s['OLD_all_pass']) & s['NEW_all_pass']).sum()
        old_pass_new_fail = (s['OLD_all_pass'] & (~s['NEW_all_pass'])).sum()
        both_fail = ((~s['OLD_all_pass']) & (~s['NEW_all_pass'])).sum()
        transition_rows.append({
            'comparison_type': 'lever_effect(OLD_vs_NEW)_candidate_level', 'fixed_axis': f'model={model}',
            'anchor_type': a, 'n': n, 'both_pass': both_pass, 'both_fail': both_fail,
            'A_fail_B_pass(improved)': old_fail_new_pass, 'A_pass_B_fail(worsened)': old_pass_new_fail,
        })

sel_pivot = sel_df.pivot_table(index=['anchor_type', 'seed_id', 'call_id'], columns='condition',
                                values='batch_success', aggfunc='first')
for lv, condA, condB in [('OLD', 'A', 'D'), ('NEW', 'B', 'C')]:
    for a in ANCHORS + ['전체']:
        s = sel_pivot if a == '전체' else sel_pivot.loc[sel_pivot.index.get_level_values('anchor_type') == a]
        n = len(s)
        both_pass = ((s[condA]) & (s[condB])).sum()
        both_fail = ((~s[condA]) & (~s[condB])).sum()
        base_fail_rl_pass = ((~s[condA]) & (s[condB])).sum()
        base_pass_rl_fail = ((s[condA]) & (~s[condB])).sum()
        transition_rows.append({
            'comparison_type': 'model_effect(BASE_vs_RL)_batch_level', 'fixed_axis': f'lever={lv}',
            'anchor_type': a, 'n': n, 'both_pass': both_pass, 'both_fail': both_fail,
            'A_fail_B_pass(improved)': base_fail_rl_pass, 'A_pass_B_fail(worsened)': base_pass_rl_fail,
        })
transitions_df = pd.DataFrame(transition_rows)
transitions_csv = os.path.join(OUT_DIR, 'track1_paired_transitions.csv')
transitions_df.to_csv(transitions_csv, index=False, encoding='utf-8-sig')
print(f'\n저장: {transitions_csv}')

# 핵심 비교: production(C) vs BASE에 최종레버만 적용(B) — RL 채택여부 핵심 질문
key_pivot = sel_pivot
for a in ANCHORS + ['전체']:
    s = key_pivot if a == '전체' else key_pivot.loc[key_pivot.index.get_level_values('anchor_type') == a]
    n = len(s)
    b_pass, c_pass = s['B'], s['C']
    print(f"  [B(BASE+NEW) vs C(RL+NEW) @ {a}] n={n} B->C 개선={((~b_pass)&c_pass).sum()} "
          f"B->C 악화={(b_pass&(~c_pass)).sum()} 둘다통과={(b_pass&c_pass).sum()} 둘다실패={((~b_pass)&(~c_pass)).sum()}")

# 통계검정: B vs C 배치성공률 차이(McNemar, paired binary) — scipy.stats로 근사(정확검정 직접 계산)
def mcnemar_p(b_only, c_only):
    n_disc = b_only + c_only
    if n_disc == 0:
        return 1.0
    stat = (abs(b_only - c_only) - 1) ** 2 / n_disc  # 연속성 보정
    return 1 - stats.chi2.cdf(stat, df=1)

for a in ANCHORS + ['전체']:
    s = key_pivot if a == '전체' else key_pivot.loc[key_pivot.index.get_level_values('anchor_type') == a]
    b_pass, c_pass = s['B'], s['C']
    b_only = (b_pass & (~c_pass)).sum(); c_only = ((~b_pass) & c_pass).sum()
    p = mcnemar_p(b_only, c_only)
    diff = c_pass.mean() - b_pass.mean()
    print(f"  [McNemar B vs C @ {a}] B_only={b_only} C_only={c_only} p={p:.4f} 성공률차이(C-B)={diff*100:.1f}%p")


# ============================================================
# 5. 다양성 비교 전용 CSV — track1_diversity_comparison.csv
# ============================================================
div_rows = []
for cond in ('A', 'B', 'C', 'D'):
    model, lv = CONDITIONS[cond]
    for a in ANCHORS + ['전체']:
        s = sel_df[(sel_df['condition'] == cond)] if a == '전체' else sel_df[(sel_df['condition'] == cond) & (sel_df['anchor_type'] == a)]
        if len(s) == 0:
            continue
        vc = s['sel_final_menus'].value_counts(normalize=True)
        vc_soup = s['sel_final_menus'].apply(lambda x: x.split('|')[1] if len(x.split('|')) > 1 else None).value_counts()
        vc_main = s['sel_final_menus'].apply(lambda x: x.split('|')[2] if len(x.split('|')) > 2 else None).value_counts()
        vc_side = s['sel_final_menus'].apply(lambda x: x.split('|')[3] if len(x.split('|')) > 3 else None).value_counts()
        vc_kim = s['sel_final_menus'].apply(lambda x: x.split('|')[-1]).value_counts()
        repeat_rate = 1 - (s['sel_final_menus'].nunique() / len(s))
        div_rows.append({
            'condition': cond, 'model': model, 'lever': lv, 'anchor_type': a, 'n_batches': len(s),
            'unique_combo': s['sel_final_menus'].nunique(), 'unique_soup': vc_soup.shape[0],
            'unique_main': vc_main.shape[0], 'unique_side': vc_side.shape[0], 'unique_kimchi': vc_kim.shape[0],
            'top1_combo_share': vc.iloc[0], 'top5_combo_share': vc.iloc[:5].sum(),
            'top1_main_share': (vc_main.iloc[0] / len(s)) if len(vc_main) else None,
            'combo_repeat_rate': repeat_rate,
        })
diversity_df = pd.DataFrame(div_rows)
diversity_csv = os.path.join(OUT_DIR, 'track1_diversity_comparison.csv')
diversity_df.to_csv(diversity_csv, index=False, encoding='utf-8-sig')
print(f'\n저장: {diversity_csv}')


# ============================================================
# 6. 대표 사례 — track1_representative_cases.csv
#    핵심축: B(BASE+최종레버) vs C(RL+최종레버, 실제 production) — RL 채택여부 핵심비교
# ============================================================
bc = sel_df[sel_df['condition'].isin(['B', 'C'])].pivot(index=['anchor_type', 'seed_id', 'call_id'], columns='condition',
                                                          values=['batch_success', 'sel_final_menus', 'sel_menus_raw',
                                                                  'sel_all_pass', 'rank_of_first_pass'])
bc.columns = [f'{a}_{b}' for a, b in bc.columns]
bc = bc.reset_index()

improved = bc[(~bc['batch_success_B']) & (bc['batch_success_C'])]
worsened = bc[(bc['batch_success_B']) & (~bc['batch_success_C'])]
both_pass_diff_menu = bc[(bc['batch_success_B']) & (bc['batch_success_C']) & (bc['sel_final_menus_B'] != bc['sel_final_menus_C'])]
identical = bc[(bc['sel_final_menus_B'] == bc['sel_final_menus_C']) & (bc['batch_success_B'] == bc['batch_success_C'])]

case_rows = []
for label, sub, n in [('B실패_C통과(개선)', improved, 5), ('B통과_C실패(악화)', worsened, 5),
                       ('둘다통과_메뉴다름', both_pass_diff_menu, 5), ('완전동일', identical, 5)]:
    for _, r in sub.head(n).iterrows():
        case_rows.append({
            'category': label, 'anchor_type': r['anchor_type'], 'seed_id': r['seed_id'], 'call_id': r['call_id'],
            'B_final_menus': r['sel_final_menus_B'], 'C_final_menus': r['sel_final_menus_C'],
            'B_batch_success': r['batch_success_B'], 'C_batch_success': r['batch_success_C'],
            'B_rank': r['rank_of_first_pass_B'], 'C_rank': r['rank_of_first_pass_C'],
        })
cases_df = pd.DataFrame(case_rows)
cases_csv = os.path.join(OUT_DIR, 'track1_representative_cases.csv')
cases_df.to_csv(cases_csv, index=False, encoding='utf-8-sig')
print(f'\n저장: {cases_csv} ({len(cases_df)}건: 개선{min(len(improved),5)}+악화{min(len(worsened),5)}+'
      f'통과다름{min(len(both_pass_diff_menu),5)}+동일{min(len(identical),5)})')

# 분석 결과 재사용을 위해 피클로도 저장 (보고서 스크립트에서 재계산 없이 로드)
with open(os.path.join(OUT_DIR, 'analysis_state.pkl'), 'wb') as f:
    pickle.dump({'df': df, 'sel_df': sel_df, 'overall_df': overall_df, 'by_anchor_df': by_anchor_df,
                 'lever_effect_df': lever_effect_df, 'transitions_df': transitions_df,
                 'diversity_df': diversity_df, 'cases_df': cases_df}, f)
print('\n분석 상태 저장 완료 (report 스크립트용): analysis_state.pkl')

