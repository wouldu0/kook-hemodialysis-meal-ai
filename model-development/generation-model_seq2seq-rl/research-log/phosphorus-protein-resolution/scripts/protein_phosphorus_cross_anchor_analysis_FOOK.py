# -*- coding: utf-8 -*-
"""
protein_phosphorus_cross_anchor_analysis_FOOK.py — cross_anchor_trace.csv를 pandas로 재분석.

★ 버그 수정 노트: 원본 생성 스크립트의 'selected'/'fallback_used' 컬럼은 candidate_id가
앵커마다 1부터 다시 시작하는데(생선구이 1~2400, 육류 1~2400) 그 값 자체만으로 전역 set에
넣어 판정해, 서로 다른 앵커의 같은 번호끼리 오염되는 결함이 있었다(예: 생선구이 5번이
선택되면 육류 5번도 '선택됨'으로 잘못 표시됨) — zero_candidate_rate=0%인데 최종선택
5영양통과율이 63~70%로 나오는 모순으로 발견됨. 이 스크립트에서는 raw 영양값(calories/
protein/potassium/phosphorus_raw/sodium_season)으로 5개 판정을 그 자리에서 재계산하고,
(anchor_type, seed, call_id) 그룹 안에서 selection을 처음부터 다시 계산한다(모델 재실행
없음, candidate_id 순서를 원 생성순서 프록시로 사용).
"""
import os
import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CODE, 'protein_phosphorus_cross_anchor_out')
TRACE_CSV = os.path.join(OUT_DIR, 'protein_phosphorus_cross_anchor_trace.csv')

ELO, EHI = 600.0, 700.0
PLO, PHI = 22.0, 24.0
KMAX = 1000.0
PMAX = 1000.0 / 3.0
NAMAX = 393.0


def top_shares(counter_series):
    vc = counter_series.value_counts()
    n = vc.sum()
    if n == 0:
        return 0.0, 0.0, vc
    top1 = vc.iloc[0] / n
    top3 = vc.iloc[:min(3, len(vc))].sum() / n
    return top1, top3, vc


def consecutive_same_rate(df_sorted, col):
    same = 0; total = 0
    for seed, g in df_sorted.groupby('seed'):
        g = g.sort_values('call_id')
        vals = g[col].tolist()
        for i in range(1, len(vals)):
            total += 1
            if vals[i] == vals[i - 1]:
                same += 1
    return same / total if total else None


def main():
    df = pd.read_csv(TRACE_CSV, encoding='utf-8-sig')
    print(f'로드: {len(df)}행')

    # ── 5개 영양 flag를 raw값에서 재계산(플래그 컬럼 오염과 무관하게 신뢰 가능한 값만 사용) ──
    df['f_phosphorus'] = df['phosphorus_raw'] < PMAX
    df['f_protein'] = (df['protein'] >= PLO) & (df['protein'] <= PHI)
    df['f_protein_low'] = df['protein'] < PLO
    df['f_calorie'] = (df['calories'] >= ELO) & (df['calories'] <= EHI)
    df['f_calorie_low'] = df['calories'] < ELO
    df['f_sodium'] = df['sodium_season'] <= NAMAX
    df['f_potassium'] = df['potassium'] < KMAX
    df['f_all'] = df['f_phosphorus'] & df['f_protein'] & df['f_calorie'] & df['f_sodium'] & df['f_potassium']

    # 원본 nutrition_all_pass와 대조 검증(생성스크립트 값과 재계산 값이 실제로 같은 근거인지 확인)
    mismatch = (df['f_all'] != df['nutrition_all_pass']).sum()
    print(f'검증: 재계산 f_all vs 원본 nutrition_all_pass 불일치 {mismatch}건 / {len(df)}건')

    def score_row(r):
        return int(r['f_phosphorus']) + int(r['f_protein']) + int(r['f_calorie']) + int(r['f_sodium']) + int(r['f_potassium'])
    df['score'] = df.apply(score_row, axis=1)

    anchors = df['anchor_type'].unique()
    variants = df['variant'].unique()

    # ── selection 재계산 (버그 수정) ──
    df['selected_fixed'] = False
    df['fallback_used_fixed'] = False
    for anchor in anchors:
        for variant in variants:
            sub = df[(df['anchor_type'] == anchor) & (df['variant'] == variant)]
            for (seed, call_id), g in sub.groupby(['seed', 'call_id']):
                g_sorted = g.sort_values('candidate_id')
                passing = g_sorted[g_sorted['f_all']]
                if len(passing):
                    sel_idx = passing.index[0]
                    df.loc[sel_idx, 'selected_fixed'] = True
                else:
                    best_idx = g_sorted['score'].idxmax()
                    df.loc[best_idx, 'selected_fixed'] = True
                    df.loc[best_idx, 'fallback_used_fixed'] = True

    # ── ② service_summary.csv ──
    service_rows = []
    selected_by_key = {}
    for anchor in anchors:
        adf = df[df['anchor_type'] == anchor]
        n_batches = adf.groupby(['seed', 'call_id']).ngroups
        for variant in variants:
            vdf = adf[adf['variant'] == variant]
            sel = vdf[vdf['selected_fixed']].copy()
            zero_cnt = 0
            for key, g in vdf.groupby(['seed', 'call_id']):
                if not g['f_all'].any():
                    zero_cnt += 1
            gen_success_rate = 1 - zero_cnt / n_batches

            final_pass_rate = sel['f_all'].mean()
            final_protein_low_rate = sel['f_protein_low'].mean()
            final_calorie_low_rate = sel['f_calorie_low'].mean()
            final_mean_protein = sel['protein'].mean()
            below22 = sel[sel['protein'] < PLO]
            mean_deficit = (PLO - below22['protein']).mean() if len(below22) else 0.0
            final_rawP_high_rate = (~sel['f_phosphorus']).mean()
            final_sodium_high_rate = (~sel['f_sodium']).mean()
            final_potassium_high_rate = (~sel['f_potassium']).mean()
            final_unreal_rate = sel['unrealistic_reason'].notna().mean()
            mean_adjust_ms = vdf['elapsed_sec'].mean() * 1000

            soup_top1, soup_top3, soup_vc = top_shares(sel['soup'])
            side_top1, side_top3, side_vc = top_shares(sel['side'])
            kim_top1, kim_top3, kim_vc = top_shares(sel['kimchi'])
            soup_streak = consecutive_same_rate(sel, 'soup')
            side_streak = consecutive_same_rate(sel, 'side')
            unique_meal = sel[['soup', 'main', 'side', 'kimchi']].drop_duplicates().shape[0]

            service_rows.append({
                'anchor_type': anchor, 'variant': variant, 'call_count': n_batches,
                'generation_success_rate': gen_success_rate, 'zero_candidate_rate': zero_cnt / n_batches,
                'final_nutrition_pass_rate': final_pass_rate, 'final_protein_low_rate': final_protein_low_rate,
                'final_mean_protein': final_mean_protein, 'mean_protein_deficit_when_low': mean_deficit,
                'final_calorie_low_rate': final_calorie_low_rate,
                'final_rawP_high_rate': final_rawP_high_rate, 'final_sodium_high_rate': final_sodium_high_rate,
                'final_potassium_high_rate': final_potassium_high_rate,
                'final_unrealistic_rate': final_unreal_rate,
                'unique_soup_count': soup_vc.shape[0], 'unique_side_count': side_vc.shape[0],
                'unique_kimchi_count': kim_vc.shape[0], 'unique_meal_count': unique_meal,
                'soup_top1_share': soup_top1, 'soup_top3_share': soup_top3,
                'side_top1_share': side_top1, 'side_top3_share': side_top3,
                'kimchi_top1_share': kim_top1,
                'soup_consecutive_same_rate': soup_streak, 'side_consecutive_same_rate': side_streak,
                'mean_adjust_time_ms': mean_adjust_ms,
                'fallback_used_count': int(sel['fallback_used_fixed'].sum()),
            })
            selected_by_key[(anchor, variant)] = sel

    service_df = pd.DataFrame(service_rows)
    service_df.to_csv(os.path.join(OUT_DIR, 'protein_phosphorus_cross_anchor_service_summary.csv'),
                       index=False, encoding='utf-8-sig')
    print('\n저장: service_summary.csv')
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(service_df.to_string(index=False))

    # ── ③ transition.csv + new_breaks.csv ──
    trans_rows = []
    new_break_rows = []
    for anchor in anchors:
        adf = df[df['anchor_type'] == anchor]
        piv_pass = adf.pivot(index='candidate_id', columns='variant', values='f_all')
        piv_protein_low = adf.pivot(index='candidate_id', columns='variant', values='f_protein_low')
        piv_calorie_low = adf.pivot(index='candidate_id', columns='variant', values='f_calorie_low')
        piv_p_high = adf.pivot(index='candidate_id', columns='variant', values='f_phosphorus').apply(lambda c: ~c)
        piv_na_high = adf.pivot(index='candidate_id', columns='variant', values='f_sodium').apply(lambda c: ~c)
        piv_k_high = adf.pivot(index='candidate_id', columns='variant', values='f_potassium').apply(lambda c: ~c)
        piv_unreal = adf.pivot(index='candidate_id', columns='variant', values='unrealistic_reason')

        base_pass = piv_pass['Baseline']; new_pass = piv_pass['B90+Unified']
        defs = {
            'baseline_pass_to_new_pass': (base_pass & new_pass),
            'baseline_pass_to_new_fail': (base_pass & ~new_pass),
            'baseline_fail_to_new_pass': (~base_pass & new_pass),
            'both_fail': (~base_pass & ~new_pass),
        }
        n = len(base_pass)
        for name, mask in defs.items():
            cnt = int(mask.sum())
            trans_rows.append({'anchor_type': anchor, 'transition_type': name, 'count': cnt, 'rate': cnt / n})

        broke_mask = base_pass & ~new_pass
        n_base_pass = int(base_pass.sum())
        trans_rows.append({'anchor_type': anchor, 'transition_type': 'baseline_normal_new_break_rate',
                            'count': int(broke_mask.sum()),
                            'rate': (broke_mask.sum() / n_base_pass) if n_base_pass else 0.0})

        for cid in piv_pass.index[broke_mask]:
            reasons = []
            if piv_protein_low.loc[cid, 'B90+Unified']: reasons.append('protein_low')
            if piv_calorie_low.loc[cid, 'B90+Unified']: reasons.append('calorie_low')
            if piv_p_high.loc[cid, 'B90+Unified']: reasons.append('phosphorus_high')
            if piv_na_high.loc[cid, 'B90+Unified']: reasons.append('sodium_high')
            if piv_k_high.loc[cid, 'B90+Unified']: reasons.append('potassium_high')
            if pd.notna(piv_unreal.loc[cid, 'B90+Unified']): reasons.append('unrealistic_amount')
            row_b = adf[(adf['candidate_id'] == cid) & (adf['variant'] == 'Baseline')].iloc[0]
            row_n = adf[(adf['candidate_id'] == cid) & (adf['variant'] == 'B90+Unified')].iloc[0]
            new_break_rows.append({
                'anchor_type': anchor, 'candidate_id': cid, 'seed': row_b['seed'], 'call_id': row_b['call_id'],
                'soup': row_b['soup'], 'main': row_b['main'], 'side': row_b['side'], 'kimchi': row_b['kimchi'],
                'baseline_protein': row_b['protein'], 'new_protein': row_n['protein'],
                'baseline_phosphorus_raw': row_b['phosphorus_raw'], 'new_phosphorus_raw': row_n['phosphorus_raw'],
                'new_failure_reasons': '+'.join(reasons) if reasons else 'unknown',
            })

    trans_df = pd.DataFrame(trans_rows)
    trans_df.to_csv(os.path.join(OUT_DIR, 'protein_phosphorus_cross_anchor_transition.csv'),
                     index=False, encoding='utf-8-sig')
    print('\n저장: transition.csv')
    print(trans_df.to_string(index=False))

    newbreak_df = pd.DataFrame(new_break_rows)
    cols = ['anchor_type', 'candidate_id', 'seed', 'call_id', 'soup', 'main', 'side', 'kimchi',
            'baseline_protein', 'new_protein', 'baseline_phosphorus_raw', 'new_phosphorus_raw', 'new_failure_reasons']
    if len(newbreak_df) == 0:
        newbreak_df = pd.DataFrame(columns=cols)
    newbreak_df.to_csv(os.path.join(OUT_DIR, 'protein_phosphorus_cross_anchor_new_breaks.csv'),
                        index=False, encoding='utf-8-sig')
    print(f'저장: new_breaks.csv ({len(newbreak_df)}건)')

    # ── ④ menu_frequency.csv ──
    freq_rows = []
    for anchor in anchors:
        for variant in variants:
            sel = selected_by_key[(anchor, variant)]
            for slot in ['soup', 'side', 'kimchi']:
                vc = sel[slot].value_counts()
                n = vc.sum()
                for rank, (menu, cnt) in enumerate(vc.items(), start=1):
                    freq_rows.append({'anchor_type': anchor, 'variant': variant, 'slot': slot, 'menu': menu,
                                       'selected_count': int(cnt), 'selected_share': cnt / n, 'rank': rank})
    freq_df = pd.DataFrame(freq_rows)
    freq_df.to_csv(os.path.join(OUT_DIR, 'protein_phosphorus_cross_anchor_menu_frequency.csv'),
                   index=False, encoding='utf-8-sig')
    print(f'저장: menu_frequency.csv ({len(freq_df)}행)')

    # ── 국 다양성 감소 원인 분해 ──
    root_cause_lines = []

    def w(*a):
        root_cause_lines.append(' '.join(str(x) for x in a))

    for anchor in anchors:
        adf = df[df['anchor_type'] == anchor]
        base_sel = selected_by_key[(anchor, 'Baseline')]
        new_sel = selected_by_key[(anchor, 'B90+Unified')]
        base_soups = set(base_sel['soup'].unique())
        new_soups = set(new_sel['soup'].unique())
        lost = base_soups - new_soups
        gained = new_soups - base_soups
        w(f'--- {anchor} ---')
        w(f'Baseline 선택된 고유국: {len(base_soups)}종, B90+Unified: {len(new_soups)}종')
        w(f'사라진 국({len(lost)}): {sorted(lost)}')
        w(f'신규 등장 국({len(gained)}): {sorted(gained)}')
        for soup in sorted(lost):
            base_variant_rows = adf[(adf['variant'] == 'Baseline') & (adf['soup'] == soup)]
            gen_cnt = len(base_variant_rows)
            gate_pass_cnt = int(base_variant_rows['f_all'].sum())
            final_sel_cnt = int((base_sel['soup'] == soup).sum())
            new_variant_rows = adf[(adf['variant'] == 'B90+Unified') & (adf['soup'] == soup)]
            new_gate_pass_cnt = int(new_variant_rows['f_all'].sum())
            new_gen_cnt = len(new_variant_rows)
            w(f'  [{soup}] Baseline: 동일후보군{gen_cnt}건 중 게이트통과{gate_pass_cnt}건, 최종선택{final_sel_cnt}건 '
              f'| B90+Unified: 동일후보군 중 게이트통과{new_gate_pass_cnt}/{new_gen_cnt}건, 최종선택0건')

    with open(os.path.join(OUT_DIR, 'soup_diversity_root_cause.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(root_cause_lines))
    print('\n저장: soup_diversity_root_cause.txt')
    print('\n'.join(root_cause_lines))

    return service_df, trans_df, newbreak_df, freq_df


if __name__ == '__main__':
    main()
