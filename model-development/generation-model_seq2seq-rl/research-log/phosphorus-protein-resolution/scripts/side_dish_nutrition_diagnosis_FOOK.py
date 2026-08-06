# -*- coding: utf-8 -*-
"""
side_dish_nutrition_diagnosis_FOOK.py — nutrition_all_pass 게이트 탈락 원인을 영양소별로 분해.

기존 side_dish_diagnosis_FOOK.py가 만든 side_dish_candidate_trace.csv(10,800행,
calories/protein/sodium/potassium/phosphorus/nutrition_all_pass 포함)를 그대로 재활용한
pandas 전용 2차 분석. 새 모델 실행 없음. 코드/기준 수정 없음(진단만).

기준값 확인(app_core_FOOK.py:348 passes(t,b), FOOK_adjust_levers.py:1163 meal_bounds(60)):
  weight=60kg, 첫 끼(consumed=None) 고정 → Elo=600 Ehi=700 Plo=22 Phi=24
  Kmax=1000(<) Pmax=333.333...(<) Namax=393(<=)
  경계: 열량·단백질·나트륨은 상하한 포함(<=), 칼륨·인은 상한 미포함(<) — 즉 정확히
  1000.0/333.33...도 실패로 카운트한다. tolerance/EPS 없음(day_ok()에만 있고 passes()엔 없음).

실행:
  python side_dish_nutrition_diagnosis_FOOK.py
  (pandas/numpy/scipy만 필요, TF/모델 로딩 불필요 — 시스템 python으로 충분)
"""
import os
import numpy as np
import pandas as pd
from itertools import combinations

CODE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(CODE, 'side_dish_diagnosis_out')
OUT_DIR = os.path.join(CODE, 'side_dish_nutrition_diagnosis_out')
os.makedirs(OUT_DIR, exist_ok=True)

TRACE_CSV = os.path.join(IN_DIR, 'side_dish_candidate_trace.csv')

# ── 기준값 (F.meal_bounds(60), consumed=None, 코드에서 확인한 실제값) ──
ELO, EHI = 600.0, 700.0
PLO, PHI = 22.0, 24.0
KMAX = 1000.0
PMAX = 1000.0 / 3.0   # 333.333...
NAMAX = 393.0

NUTRIENTS5 = ['calorie', 'protein', 'sodium', 'potassium', 'phosphorus']


def entropy_of(counts):
    vals = np.array(list(counts), dtype=float)
    total = vals.sum()
    if total == 0:
        return 0.0
    p = vals / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def stage_stats_from_series(s):
    """value_counts 기반 unique/entropy/top1_ratio."""
    vc = s.value_counts()
    n = int(vc.sum())
    if n == 0:
        return {'n': 0, 'unique': 0, 'entropy': 0.0, 'top1_ratio': 0.0}
    return {'n': n, 'unique': int(vc.shape[0]), 'entropy': entropy_of(vc.values),
            'top1_ratio': float(vc.iloc[0] / n)}


def main():
    df = pd.read_csv(TRACE_CSV, encoding='utf-8-sig')
    n_total = len(df)
    print(f'로드: {TRACE_CSV} ({n_total}행)')

    # ────────────────────────────────────────────────────────────
    # 1) 후보별 영양 실패 분해 (7 boolean + 파생 컬럼)
    # ────────────────────────────────────────────────────────────
    df['calorie_low_fail'] = df['calories'] < ELO
    df['calorie_high_fail'] = df['calories'] > EHI
    df['protein_low_fail'] = df['protein'] < PLO
    df['protein_high_fail'] = df['protein'] > PHI
    df['sodium_high_fail'] = df['sodium'] > NAMAX
    df['potassium_high_fail'] = df['potassium'] >= KMAX
    df['phosphorus_high_fail'] = df['phosphorus'] >= PMAX

    fail_bool_cols = ['calorie_low_fail', 'calorie_high_fail', 'protein_low_fail', 'protein_high_fail',
                       'sodium_high_fail', 'potassium_high_fail', 'phosphorus_high_fail']
    df['nutrition_fail_count'] = df[fail_bool_cols].sum(axis=1)

    # 검증: 재계산한 fail_count==0 이 원본 nutrition_all_pass 컬럼과 일치하는지 확인 (임의 판단 금지 원칙)
    recomputed_pass = df['nutrition_fail_count'] == 0
    mismatch = (recomputed_pass != df['nutrition_all_pass']).sum()
    print(f'검증: 재계산 pass vs 원본 nutrition_all_pass 불일치 {mismatch}건 / {n_total}건')

    df['single_fail'] = df['nutrition_fail_count'] == 1
    df['multi_fail'] = df['nutrition_fail_count'] >= 2

    nutrient_fail_flag = pd.DataFrame({
        'calorie': df['calorie_low_fail'] | df['calorie_high_fail'],
        'protein': df['protein_low_fail'] | df['protein_high_fail'],
        'sodium': df['sodium_high_fail'],
        'potassium': df['potassium_high_fail'],
        'phosphorus': df['phosphorus_high_fail'],
    })

    def signature(row):
        parts = []
        if row['calorie_low_fail']: parts.append('calorie_low')
        if row['calorie_high_fail']: parts.append('calorie_high')
        if row['protein_low_fail']: parts.append('protein_low')
        if row['protein_high_fail']: parts.append('protein_high')
        if row['sodium_high_fail']: parts.append('sodium_high')
        if row['potassium_high_fail']: parts.append('potassium_high')
        if row['phosphorus_high_fail']: parts.append('phosphorus_high')
        return '+'.join(parts) if parts else 'pass'

    df['nutrition_fail_signature'] = df.apply(signature, axis=1)

    detail_cols = ['anchor_type', 'seed_id', 'call_id', 'candidate_id', 'side_dish', 'adjusted_side_dish',
                   'calories', 'protein', 'sodium', 'potassium', 'phosphorus',
                   'calorie_low_fail', 'calorie_high_fail', 'protein_low_fail', 'protein_high_fail',
                   'sodium_high_fail', 'potassium_high_fail', 'phosphorus_high_fail',
                   'nutrition_fail_count', 'nutrition_fail_signature', 'nutrition_all_pass']
    detail_csv = os.path.join(OUT_DIR, 'side_dish_nutrition_fail_detail.csv')
    df[detail_cols].to_csv(detail_csv, index=False, encoding='utf-8-sig')
    print(f'① {detail_csv} ({len(df)}행)')

    # ────────────────────────────────────────────────────────────
    # 2) 전체/앵커별 영양소 실패 요약 + counterfactual
    # ────────────────────────────────────────────────────────────
    def nutrient_summary_for_scope(sub, nff_sub, scope_name, anchor_name):
        rows = []
        n_sub = len(sub)
        fail_sub = ~sub['nutrition_all_pass']
        n_fail = int(fail_sub.sum())
        base_pass_c = sub.loc[sub['nutrition_all_pass'], 'adjusted_side_dish']
        base_stats = stage_stats_from_series(base_pass_c)

        for nut in NUTRIENTS5:
            included = nff_sub[nut]
            included_fail_count = int(included.sum())
            included_fail_rate = included_fail_count / n_sub if n_sub else np.nan

            # 단독 실패: 그 영양소만 실패(다른 4개 영양소는 통과)
            others = nff_sub.drop(columns=[nut])
            single_this = included & (~others.any(axis=1))
            single_fail_count = int(single_this.sum())
            single_fail_rate = single_fail_count / n_fail if n_fail else np.nan  # 실패후보 중 비율
            single_fail_rate_of_all = single_fail_count / n_sub if n_sub else np.nan  # 전체후보 중 비율

            # counterfactual: 이 영양소 기준만 무시 → 구제되는 후보 = "이 영양소만 단독으로 실패"한 후보
            rescued = single_this  # 다른 영양소가 이미 실패면 이 영양소 기준을 없애도 여전히 실패
            rescued_count = int(rescued.sum())
            pass_gain = rescued_count / n_sub if n_sub else np.nan

            cf_pass_dish = pd.concat([base_pass_c, sub.loc[rescued, 'adjusted_side_dish']])
            cf_stats = stage_stats_from_series(cf_pass_dish)
            unique_gain = cf_stats['unique'] - base_stats['unique']
            entropy_gain = cf_stats['entropy'] - base_stats['entropy']
            top1_change = cf_stats['top1_ratio'] - base_stats['top1_ratio']

            rows.append({
                'scope': scope_name, 'anchor_type': anchor_name, 'nutrient': nut,
                'candidate_count': n_sub, 'fail_count_total': n_fail,
                'included_fail_count': included_fail_count, 'included_fail_rate': included_fail_rate,
                'single_fail_count': single_fail_count,
                'single_fail_rate_of_failures': single_fail_rate,
                'single_fail_rate_of_all': single_fail_rate_of_all,
                'counterfactual_rescued_count': rescued_count,
                'counterfactual_pass_gain': pass_gain,
                'pass_rate_before': base_stats['n'] / n_sub if n_sub else np.nan,
                'pass_rate_after_cf': cf_stats['n'] / n_sub if n_sub else np.nan,
                'unique_side_before': base_stats['unique'], 'unique_side_after_cf': cf_stats['unique'],
                'unique_side_gain': unique_gain,
                'entropy_before': base_stats['entropy'], 'entropy_after_cf': cf_stats['entropy'],
                'entropy_gain': entropy_gain,
                'top1_ratio_before': base_stats['top1_ratio'], 'top1_ratio_after_cf': cf_stats['top1_ratio'],
                'top1_ratio_change': top1_change,
            })
        return rows

    summary_rows = []
    summary_rows += nutrient_summary_for_scope(df, nutrient_fail_flag, 'overall', 'ALL')
    for anchor in df['anchor_type'].unique():
        mask = df['anchor_type'] == anchor
        summary_rows += nutrient_summary_for_scope(df[mask].reset_index(drop=True),
                                                     nutrient_fail_flag[mask.values].reset_index(drop=True),
                                                     'anchor', anchor)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUT_DIR, 'side_dish_nutrient_failure_summary.csv')
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    print(f'② {summary_csv} ({len(summary_df)}행)')

    # single/multi-fail 비율, 평균 fail_count (scope별)
    overview_rows = []
    for scope_name, sub in [('overall', df)] + [('anchor:' + a, df[df['anchor_type'] == a]) for a in df['anchor_type'].unique()]:
        n_sub = len(sub)
        n_fail = int((~sub['nutrition_all_pass']).sum())
        overview_rows.append({
            'scope': scope_name, 'n': n_sub,
            'nutrition_pass_rate': (sub['nutrition_all_pass']).mean(),
            'nutrition_fail_rate': 1 - (sub['nutrition_all_pass']).mean(),
            'single_fail_rate_of_all': sub['single_fail'].mean(),
            'multi_fail_rate_of_all': sub['multi_fail'].mean(),
            'single_fail_rate_of_failures': sub.loc[~sub['nutrition_all_pass'], 'single_fail'].mean() if n_fail else np.nan,
            'multi_fail_rate_of_failures': sub.loc[~sub['nutrition_all_pass'], 'multi_fail'].mean() if n_fail else np.nan,
            'mean_fail_count_all': sub['nutrition_fail_count'].mean(),
            'mean_fail_count_among_failures': sub.loc[~sub['nutrition_all_pass'], 'nutrition_fail_count'].mean() if n_fail else np.nan,
        })
    overview_df = pd.DataFrame(overview_rows)
    overview_csv = os.path.join(OUT_DIR, 'side_dish_nutrition_overview.csv')
    overview_df.to_csv(overview_csv, index=False, encoding='utf-8-sig')
    print(f'②b {overview_csv}')

    # ────────────────────────────────────────────────────────────
    # 3) 실패 조합(signature) 분포 — 전체 Top-10 + 앵커별
    # ────────────────────────────────────────────────────────────
    combo_rows = []
    for scope_name, sub in [('ALL', df)] + [(a, df[df['anchor_type'] == a]) for a in df['anchor_type'].unique()]:
        fails = sub[~sub['nutrition_all_pass']]
        n_fail = len(fails)
        if n_fail == 0:
            continue
        vc = fails['nutrition_fail_signature'].value_counts()
        for sig, cnt in vc.items():
            rows_sig = fails[fails['nutrition_fail_signature'] == sig]
            uniq = rows_sig['adjusted_side_dish'].nunique()
            top_dishes = rows_sig['adjusted_side_dish'].value_counts().head(5)
            top_str = '; '.join(f'{m}({c})' for m, c in top_dishes.items())
            combo_rows.append({
                'anchor_type': scope_name, 'failure_signature': sig, 'count': int(cnt),
                'ratio': cnt / n_fail, 'unique_side_count': uniq, 'top_side_dishes': top_str,
            })
    combo_df = pd.DataFrame(combo_rows)
    combo_csv = os.path.join(OUT_DIR, 'side_dish_failure_combinations.csv')
    combo_df.to_csv(combo_csv, index=False, encoding='utf-8-sig')
    print(f'③ {combo_csv} ({len(combo_df)}행)')

    # ────────────────────────────────────────────────────────────
    # 4) 부찬 메뉴별 영양 프로파일 (adjusted_side_dish 기준 = 실제 게이트 평가 대상)
    # ────────────────────────────────────────────────────────────
    gen_count = df['side_dish'].value_counts()  # S0 생성 시점(조정 전) 카운트
    menu_rows = []
    for menu, grp in df.groupby('adjusted_side_dish'):
        n_gate = len(grp)
        n_pass = int(grp['nutrition_all_pass'].sum())
        fails = grp[~grp['nutrition_all_pass']]
        main_fail_nut = None
        if len(fails):
            nff_grp = nutrient_fail_flag.loc[grp.index]
            nff_fail = nff_grp.loc[fails.index]
            counts = nff_fail.sum().sort_values(ascending=False)
            if counts.iloc[0] > 0:
                main_fail_nut = counts.index[0]
        main_fail_sig = fails['nutrition_fail_signature'].value_counts().index[0] if len(fails) else None

        menu_rows.append({
            'side_dish': menu,
            'generated_count': int(gen_count.get(menu, 0)),
            'nutrition_gate_count': n_gate,
            'nutrition_pass_count': n_pass,
            'nutrition_pass_rate': n_pass / n_gate if n_gate else np.nan,
            'main_failure_nutrient': main_fail_nut,
            'main_failure_signature': main_fail_sig,
            'mean_calories': grp['calories'].mean(), 'mean_protein': grp['protein'].mean(),
            'mean_sodium': grp['sodium'].mean(), 'mean_potassium': grp['potassium'].mean(),
            'mean_phosphorus': grp['phosphorus'].mean(),
            'mean_fail_count': grp['nutrition_fail_count'].mean(),
            'single_fail_ratio': grp['single_fail'].mean(),
            'multi_fail_ratio': grp['multi_fail'].mean(),
        })
    menu_df = pd.DataFrame(menu_rows).sort_values('generated_count', ascending=False)
    menu_csv = os.path.join(OUT_DIR, 'side_dish_menu_nutrition_profile.csv')
    menu_df.to_csv(menu_csv, index=False, encoding='utf-8-sig')
    print(f'④ {menu_csv} ({len(menu_df)}종)')

    # ────────────────────────────────────────────────────────────
    # 5) 경계값 민감도(threshold distance) — 실패 후보의 초과/미달량 분포
    # ────────────────────────────────────────────────────────────
    def bucket_mg(x):
        if x <= 50: return '+1~50'
        if x <= 100: return '+51~100'
        if x <= 200: return '+101~200'
        return '+200 초과'

    def bucket_pct(dist_ratio):
        # dist_ratio = |초과or미달량| / 기준값
        if dist_ratio <= 0.01: return '기준선 1% 이내'
        if dist_ratio <= 0.05: return '기준선 5% 이내'
        if dist_ratio <= 0.10: return '기준선 10% 이내'
        return '기준선 10% 초과(구조적)'

    dist_specs = [
        ('sodium_high', df['sodium'] - NAMAX, NAMAX, df['sodium_high_fail']),
        ('potassium_high', df['potassium'] - KMAX, KMAX, df['potassium_high_fail']),
        ('phosphorus_high', df['phosphorus'] - PMAX, PMAX, df['phosphorus_high_fail']),
        ('protein_low', PLO - df['protein'], PLO, df['protein_low_fail']),
        ('calorie_low', ELO - df['calories'], ELO, df['calorie_low_fail']),
        ('protein_high', df['protein'] - PHI, PHI, df['protein_high_fail']),
        ('calorie_high', df['calories'] - EHI, EHI, df['calorie_high_fail']),
    ]
    dist_rows = []
    for name, dist_series, thr, failmask in dist_specs:
        sub_dist = dist_series[failmask]
        n_fail_nut = len(sub_dist)
        if n_fail_nut == 0:
            continue
        # 버킷명은 mg 기준이지만 단백질(g)·열량(kcal)에도 동일 구간폭을 재사용해 절대편차 크기만 구분
        buckets = sub_dist.apply(bucket_mg)
        vc = buckets.value_counts()
        pct_bucket = (sub_dist.abs() / thr).apply(bucket_pct)
        pct_vc = pct_bucket.value_counts()
        top_dishes = df.loc[sub_dist.index, 'adjusted_side_dish'].value_counts().head(5)
        for b, c in vc.items():
            dist_rows.append({
                'nutrient': name, 'distance_bucket': b, 'candidate_count': int(c),
                'candidate_ratio': c / n_fail_nut, 'unique_side_count': None, 'top_side_dishes': None,
            })
        for b, c in pct_vc.items():
            dist_rows.append({
                'nutrient': name, 'distance_bucket': b, 'candidate_count': int(c),
                'candidate_ratio': c / n_fail_nut, 'unique_side_count': None, 'top_side_dishes': None,
            })
        dist_rows.append({
            'nutrient': name, 'distance_bucket': 'ALL_FAILS(참고)', 'candidate_count': n_fail_nut,
            'candidate_ratio': 1.0, 'unique_side_count': int(df.loc[sub_dist.index, 'adjusted_side_dish'].nunique()),
            'top_side_dishes': '; '.join(f'{m}({c})' for m, c in top_dishes.items()),
        })
    dist_df = pd.DataFrame(dist_rows)
    dist_csv = os.path.join(OUT_DIR, 'side_dish_threshold_distance.csv')
    dist_df.to_csv(dist_csv, index=False, encoding='utf-8-sig')
    print(f'⑤ {dist_csv} ({len(dist_df)}행)')

    # ────────────────────────────────────────────────────────────
    # 6) 영양소 간 상호작용 (phi coefficient / Jaccard, 5-영양 실패 플래그 기준)
    # ────────────────────────────────────────────────────────────
    inter_rows = []
    for a, b_ in combinations(NUTRIENTS5, 2):
        fa = nutrient_fail_flag[a].values
        fb = nutrient_fail_flag[b_].values
        n11 = int((fa & fb).sum()); n10 = int((fa & ~fb).sum())
        n01 = int((~fa & fb).sum()); n00 = int((~fa & ~fb).sum())
        n1x = n11 + n10; n0x = n01 + n00; nx1 = n11 + n01; nx0 = n10 + n00
        denom = np.sqrt(n1x * n0x * nx1 * nx0)
        phi = (n11 * n00 - n10 * n01) / denom if denom > 0 else np.nan
        union = (fa | fb).sum()
        jaccard = n11 / union if union > 0 else np.nan
        inter_rows.append({
            'nutrient_a': a, 'nutrient_b': b_, 'both_fail_count': n11,
            'a_only_count': n10, 'b_only_count': n01, 'neither_fail_count': n00,
            'phi_coefficient': phi, 'jaccard_similarity': jaccard,
            'cooccur_ratio_of_total': n11 / n_total,
        })
    inter_df = pd.DataFrame(inter_rows).sort_values('phi_coefficient', ascending=False)
    inter_csv = os.path.join(OUT_DIR, 'side_dish_nutrient_interaction.csv')
    inter_df.to_csv(inter_csv, index=False, encoding='utf-8-sig')
    print(f'⑥ {inter_csv} ({len(inter_df)}행)')

    # ────────────────────────────────────────────────────────────
    # 7) 조합문제 proxy: 같은 부찬(adjusted_side_dish)이 앵커별로 통과율이 다른가
    #    (진짜 슬롯분해 데이터 없음 → 대리지표로만 사용, §보고서에 한계 명시)
    # ────────────────────────────────────────────────────────────
    cross_rows = []
    pivot = df.groupby(['adjusted_side_dish', 'anchor_type'])['nutrition_all_pass'].agg(['mean', 'count'])
    for menu, grp in df.groupby('adjusted_side_dish'):
        anchors_present = grp['anchor_type'].unique()
        if len(anchors_present) < 2:
            continue
        rates = grp.groupby('anchor_type')['nutrition_all_pass'].mean()
        counts = grp.groupby('anchor_type')['nutrition_all_pass'].count()
        if (counts < 5).any():
            continue  # 표본 너무 작은 건 변동성 판단에서 제외
        cross_rows.append({
            'side_dish': menu, 'n_anchors': len(anchors_present),
            'pass_rate_min': rates.min(), 'pass_rate_max': rates.max(),
            'pass_rate_range': rates.max() - rates.min(),
            'pass_rate_by_anchor': '; '.join(f'{a}={rates[a]:.2f}(n={counts[a]})' for a in rates.index),
        })
    cross_df = pd.DataFrame(cross_rows).sort_values('pass_rate_range', ascending=False)
    cross_csv = os.path.join(OUT_DIR, 'side_dish_anchor_combination_proxy.csv')
    cross_df.to_csv(cross_csv, index=False, encoding='utf-8-sig')
    print(f'⑦ {cross_csv} ({len(cross_df)}종, 표본>=5인 앵커에서만 비교)')

    print('\n===== 콘솔 요약 =====')
    print(overview_df.to_string(index=False))
    print('\n영양소별 포함 실패율(overall):')
    print(summary_df[summary_df['scope'] == 'overall'][['nutrient', 'included_fail_rate', 'single_fail_rate_of_all', 'counterfactual_pass_gain', 'entropy_gain', 'unique_side_gain']].to_string(index=False))

    return df, summary_df, overview_df, combo_df, menu_df, dist_df, inter_df, cross_df


if __name__ == '__main__':
    main()
