# -*- coding: utf-8 -*-
"""
lever_system_audit_analysis_FOOK.py — lever_interaction_step_trace.csv(2,400건x16스텝)를
pandas로 재분석: 레버별 noop/구제율/신규실패율, 12개 지정 상호작용쌍, 회귀(regression) 매트릭스.
모델/레버 재실행 없음(순수 CSV 분석).
"""
import os
import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(CODE, 'lever_system_audit_out')
OUT_DIR = IN_DIR
TRACE_CSV = os.path.join(IN_DIR, 'lever_interaction_step_trace.csv')

ELO, EHI = 600.0, 700.0
PLO, PHI = 22.0, 24.0
KMAX = 1000.0
PMAX = 1000.0 / 3.0
NAMAX = 393.0

STEP_ORDER = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9',
              'S10', 'S11', 'S12', 'S13', 'S14', 'S15']
STEP_IDX = {s: i for i, s in enumerate(STEP_ORDER)}

LEVER_TARGET = {
    'kimchi': 'sodium', 'sodium': 'sodium', 'sodium_extra': 'sodium',
    'potassium': 'potassium', 'phosphorus': 'phosphorus', 'protein': 'protein', 'calorie': 'calorie',
}


def flags_row(row):
    return {
        'phosphorus': row['phosphorus_raw'] < PMAX,
        'protein': PLO <= row['protein'] <= PHI,
        'calorie': ELO <= row['calories'] <= EHI,
        'sodium': row['sodium_season'] <= NAMAX,
        'potassium': row['potassium'] < KMAX,
    }


def main():
    df = pd.read_csv(TRACE_CSV, encoding='utf-8-sig')
    n_cand = df['candidate_id'].nunique()
    print(f'로드: {TRACE_CSV} ({len(df)}행, {n_cand}후보)')

    df['f_phosphorus'] = df['phosphorus_raw'] < PMAX
    df['f_protein'] = (df['protein'] >= PLO) & (df['protein'] <= PHI)
    df['f_calorie'] = (df['calories'] >= ELO) & (df['calories'] <= EHI)
    df['f_sodium'] = df['sodium_season'] <= NAMAX
    df['f_potassium'] = df['potassium'] < KMAX
    df['f_all'] = df['f_phosphorus'] & df['f_protein'] & df['f_calorie'] & df['f_sodium'] & df['f_potassium']

    # ★ 원본 생성스크립트의 'noop'/'changed_menu'는 (메뉴,재료) 키 기준 amt diff라서
    # 김치 통째교체(inst[:]=... 재구성)나 재료 스왑(i['ing']=rep, 같은 리스트원소의 필드만 변경)처럼
    # "키 자체가 바뀌는" 변화를 놓친다(예: 김치 noop_rate=1.000인데 potassium 페어분석에선 실제로
    # 119건이 바뀐 것으로 나와 모순 — 감지 알고리즘 결함으로 확인됨). 여기서는 실제 영양총계
    # delta(6개 값)로 noop을 재계산해 이 blind spot을 없앤다(추가 실행 없이 기존 trace의 delta
    # 컬럼만 사용, 모델/레버 재실행 없음).
    delta_cols = ['delta_calorie', 'delta_protein', 'delta_potassium', 'delta_phosphorus_raw', 'delta_sodium']
    df['noop'] = (df[delta_cols].abs() < 1e-6).all(axis=1)
    df['changed_state'] = ~df['noop']

    # wide pivot: candidate x step_order -> flags/values
    pivot = {}
    for col in ['f_phosphorus', 'f_protein', 'f_calorie', 'f_sodium', 'f_potassium', 'f_all',
                'calories', 'protein', 'potassium', 'phosphorus_raw', 'phosphorus_effective', 'sodium_season',
                'noop', 'changed_menu']:
        pivot[col] = df.pivot(index='candidate_id', columns='step_order', values=col)

    # ────────────────────────────────────────────────────────────
    # ① lever_call_summary.csv
    # ────────────────────────────────────────────────────────────
    call_rows = []
    for lever in ['kimchi', 'sodium', 'sodium_extra', 'potassium', 'phosphorus', 'protein', 'calorie']:
        sub = df[df['lever_name'] == lever]
        n = len(sub)
        target = LEVER_TARGET[lever]
        noop_rate = sub['noop'].mean()
        # target 구제율: 이 스텝의 직전 상태(같은 candidate, 직전 step_order)에서 target이 fail이었는데
        # 이 스텝 이후 pass가 된 비율 (직전상태는 changed 계산과 동일 방식으로 별도 lookup)
        sub2 = sub.copy()
        sub2['step_idx'] = sub2['step_order'].map(STEP_IDX)
        rescue_count = 0; rescue_denom = 0; break_other_count = 0
        for _, r in sub2.iterrows():
            cid = r['candidate_id']; idx = r['step_idx']
            if idx == 0:
                continue
            prev_step = STEP_ORDER[idx - 1]
            prev_flags = {nut: pivot[f'f_{nut}'].loc[cid, prev_step] for nut in
                          ['phosphorus', 'protein', 'calorie', 'sodium', 'potassium']}
            cur_flags = {'phosphorus': r['f_phosphorus'], 'protein': r['f_protein'], 'calorie': r['f_calorie'],
                         'sodium': r['f_sodium'], 'potassium': r['f_potassium']}
            if not prev_flags[target]:
                rescue_denom += 1
                if cur_flags[target]:
                    rescue_count += 1
            for nut in cur_flags:
                if nut == target:
                    continue
                if prev_flags[nut] and not cur_flags[nut]:
                    break_other_count += 1
        call_rows.append({
            'lever_name': lever, 'target_nutrient': target, 'call_count': n,
            'noop_rate': noop_rate, 'changed_rate': 1 - noop_rate,
            'target_fail_before_count': rescue_denom,
            'target_rescue_count': rescue_count,
            'target_rescue_rate': rescue_count / rescue_denom if rescue_denom else None,
            'other_nutrient_break_count': break_other_count,
            'other_nutrient_break_rate': break_other_count / n,
        })
    call_df = pd.DataFrame(call_rows)
    call_df.to_csv(os.path.join(OUT_DIR, 'lever_call_summary.csv'), index=False, encoding='utf-8-sig')
    print(f'① lever_call_summary.csv ({len(call_df)}행)')

    # ────────────────────────────────────────────────────────────
    # ② lever_pair_interaction.csv (12개 지정쌍)
    # ────────────────────────────────────────────────────────────
    # (label, before_step, after_step, nutrient_checked, recovery_step(pass2 동일지점), final_step)
    pairs = [
        ('phosphorus_lever -> protein', 'S4', 'S5', 'protein', 'S12', 'S15'),   # phosphorus 자신의 효과가 protein에
        ('phosphorus_lever -> calorie', 'S4', 'S5', 'calorie', 'S15', 'S15'),
        ('protein_lever -> phosphorus', 'S5', 'S6', 'phosphorus', 'S11', 'S15'),
        ('calorie_lever -> phosphorus', 'S8', 'S9', 'phosphorus', 'S15', 'S15'),
        ('sodium_lever_pair -> protein', 'S6', 'S8', 'protein', 'S14', 'S15'),
        ('protein_lever -> sodium', 'S5', 'S6', 'sodium', 'S13', 'S15'),
        ('potassium_lever -> phosphorus', 'S3', 'S4', 'phosphorus', 'S10', 'S15'),
        ('potassium_lever -> protein', 'S3', 'S4', 'protein', 'S10', 'S15'),
        ('kimchi_lever -> sodium', 'S0', 'S1', 'sodium', 'S1', 'S15'),
        ('kimchi_lever -> potassium', 'S0', 'S1', 'potassium', 'S1', 'S15'),
        ('calorie_lever -> sodium', 'S8', 'S9', 'sodium', 'S15', 'S15'),
        ('calorie_lever -> phosphorus(dup_of_above)', 'S8', 'S9', 'phosphorus', 'S15', 'S15'),
    ]
    pair_rows = []
    for label, before_s, after_s, nut, recov_s, final_s in pairs:
        before_col = pivot[f'f_{nut}'][before_s]
        after_col = pivot[f'f_{nut}'][after_s]
        recov_col = pivot[f'f_{nut}'][recov_s]
        final_col = pivot[f'f_{nut}'][final_s]
        fixed_by_first = before_col == False  # noqa: E712  (당시 실패 상태였던 후보 전체 모수는 아래서 별도)
        # 앞레버가 맞춘 값 = before False, after True
        fixed_count = ((~before_col) & (after_col)).sum()
        # 뒤레버가 다시 깨뜨림 = after True(방금 맞춰짐) 인데 그 다음 스텝에서 다시 False? 여기선
        # after_s 자체가 "뒤레버 직후" 이므로, "앞레버가 맞춘 걸 뒤레버가 깬다"는 별도 정의가 필요.
        # 요청 사양: "앞 레버가 맞춘 값" / "뒤 레버가 다시 깨뜨린 건수" 이므로 before_s=앞레버 직후,
        # after_s=뒤레버 직후로 해석(변수명 그대로) — 즉 "앞레버가 pass 시켜놓은 걸 뒤레버가 깨는 경우".
        broken_count = ((before_col == True) & (after_col == False)).sum()  # noqa: E712
        recovered_count = ((before_col == True) & (after_col == False) & (recov_col == True)).sum()  # noqa: E712
        still_fail_final = ((before_col == True) & (after_col == False) & (final_col == False)).sum()  # noqa: E712
        pair_rows.append({
            'pair': label, 'nutrient_checked': nut, 'before_step': before_s, 'after_step': after_s,
            'fixed_by_before_lever_count': fixed_count,
            'broken_by_after_lever_count': broken_count,
            'broken_rate_of_fixed': broken_count / before_col.sum() if before_col.sum() else None,
            'recovered_in_pass2_count': recovered_count,
            'still_failing_at_final_count': still_fail_final,
        })
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(os.path.join(OUT_DIR, 'lever_pair_interaction.csv'), index=False, encoding='utf-8-sig')
    print(f'② lever_pair_interaction.csv ({len(pair_df)}행)')

    # ────────────────────────────────────────────────────────────
    # ③ lever_noop_and_regression.csv (레버 x 영양소 회귀매트릭스 + pass별 noop)
    # ────────────────────────────────────────────────────────────
    reg_rows = []
    df['pass_number'] = df['pass_number'].astype(str)
    for lever in ['kimchi', 'sodium', 'sodium_extra', 'potassium', 'phosphorus', 'protein', 'calorie']:
        sub = df[df['lever_name'] == lever].copy()
        sub['step_idx'] = sub['step_order'].map(STEP_IDX)
        for pass_label in sub['pass_number'].unique():
            psub = sub[sub['pass_number'] == pass_label]
            noop_rate = psub['noop'].mean()
            row = {'lever_name': lever, 'pass_position': pass_label, 'call_count': len(psub),
                   'noop_rate': noop_rate}
            for nut in ['phosphorus', 'protein', 'calorie', 'sodium', 'potassium']:
                if nut == LEVER_TARGET[lever]:
                    row[f'{nut}_break_rate'] = None
                    continue
                brk = 0; denom = 0
                for _, r in psub.iterrows():
                    cid = r['candidate_id']; idx = r['step_idx']
                    if idx == 0:
                        continue
                    prev_step = STEP_ORDER[idx - 1]
                    prev_ok = pivot[f'f_{nut}'].loc[cid, prev_step]
                    cur_ok = r[f'f_{nut}']
                    if prev_ok:
                        denom += 1
                        if not cur_ok:
                            brk += 1
                row[f'{nut}_break_rate'] = brk / denom if denom else None
            reg_rows.append(row)
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(os.path.join(OUT_DIR, 'lever_noop_and_regression.csv'), index=False, encoding='utf-8-sig')
    print(f'③ lever_noop_and_regression.csv ({len(reg_df)}행)')

    # ── 콘솔 요약 ──
    print('\n=== 레버별 호출요약 ===')
    print(call_df.to_string(index=False))
    print('\n=== 12개 상호작용쌍 ===')
    print(pair_df.to_string(index=False))

    return df, call_df, pair_df, reg_df


if __name__ == '__main__':
    main()
