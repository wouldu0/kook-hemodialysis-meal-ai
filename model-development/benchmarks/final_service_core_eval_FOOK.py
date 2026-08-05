# -*- coding: utf-8 -*-
"""
final_service_core_eval_FOOK.py — 최종 FOOK production 서비스(RL i002 + 현재 최종
FOOK_adjust_levers.py, 두부·콩류 raw P 통일+90%cap 포함)의 핵심 성능만 간단 평가.

모델 단계 비교가 아니라, 사용자가 실제로 받는 "최종 선택 식단 1개"가 서비스 요구사항을
충족하는지만 본다. app_core_FOOK.make_meal()을 직접 호출해 production 전체 경로(생성→
adjust→passes→후보선택→fallback)를 그대로 실행한다 - 후보 선택 로직을 손으로 복제하지 않음.

서비스 코드/체크포인트/레버 코드는 수정하지 않는다. 평가·보고만 수행.
"""
import os, sys, time, csv, statistics
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

CODE = r'E:\final\Diet-Generation-As-Sequence-master\Diet-Generation-As-Sequence-master\Code'
sys.path.insert(0, CODE)
sys.path.insert(0, r'E:\final')
import numpy as np

OUT_DIR = os.path.join(CODE, 'final_service_core_eval_out')
os.makedirs(OUT_DIR, exist_ok=True)

print('=' * 70)
print('[1] app_core_FOOK 임포트 (production: RL i002 + 현재 최종 레버)')
print('=' * 70)
import app_core_FOOK as core   # noqa: E402
print(f'  core.CKPT = {core.CKPT}')
print(f'  core.F에 최종 레버 함수 존재(있어야 함): {hasattr(core.F, "lever_phosphorus_rawP")}')

WEIGHT = 60
b0 = core.F.meal_bounds(WEIGHT)

print('\n' + '=' * 70)
print('[2] 스모크 확인: make_meal() 직접 호출 1회')
print('=' * 70)
np.random.seed(1)
try:
    cand, note, b, anchor, warn = core.make_meal(menu='두부양념조림', W=WEIGHT)
    menus, inst, after, ok = cand
    print(f'  [OK] make_meal 직접 호출 성공: menus={menus}, ok={ok}')
except Exception as e:
    print(f'  [STOP] make_meal 호출 실패: {e}')
    import traceback; traceback.print_exc()
    sys.exit(1)

# ============================================================
# 300개 시나리오 정의: 4개 대표 앵커(두부콩류/생선구이/육류/랜덤) x 75개
# (이전 세션 전체에서 써온 동일한 4개 대표 앵커 - 두부양념조림/고등어구이/제육불고기/랜덤)
# ============================================================
SCENARIO_CONFIGS = [
    ('두부콩류', '두부양념조림', 75),
    ('생선구이', '고등어구이', 75),
    ('육류', '제육불고기', 75),
    ('랜덤', None, 75),
]
scenarios = []
for anchor_type, anchor_menu, n in SCENARIO_CONFIGS:
    for i in range(n):
        scenarios.append({'anchor_type': anchor_type, 'anchor_menu': anchor_menu, 'scenario_idx': i})
print(f'\n총 시나리오: {len(scenarios)}개 (두부콩류75+생선구이75+육류75+랜덤75)')

N_REPEATS = 3
NUT_ORDER = ['열량', '단백질', '칼륨', '인', '나트륨']


def nutrient_flags(t, bb):
    e_pass = bb['Elo'] <= t['E'] <= bb['Ehi']
    p_pass = bb['Plo'] <= t['protein'] <= bb['Phi']
    k_pass = t['K'] < bb['Kmax']
    ph_pass = t['P'] < bb['Pmax']          # raw P 기준(서비스 게이트와 동일, Peff 아님)
    na_pass = t['Na_season'] <= bb['Namax']
    return {'열량': e_pass, '단백질': p_pass, '칼륨': k_pass, '인': ph_pass, '나트륨': na_pass}


all_run_rows = []
t0 = time.perf_counter()
for rep in range(N_REPEATS):
    print(f'\n[반복 {rep + 1}/{N_REPEATS}] 300개 시나리오 실행...')
    for s in scenarios:
        seed_val = 900000 + hash((rep, s['anchor_type'], s['scenario_idx'])) % 100000
        np.random.seed(seed_val)
        t_start = time.perf_counter()
        try:
            cand, note, b, anchor, warn = core.make_meal(menu=s['anchor_menu'], W=WEIGHT)
            elapsed = time.perf_counter() - t_start
            menus, inst, after, ok = cand
            flags = nutrient_flags(after, b)
            all5_pass = all(flags.values())
            unreal = core.F.unrealistic_reason(inst) is not None
            final_menus = list(dict.fromkeys(i['menu'] for i in inst))
            anchor_preserved = (s['anchor_menu'] is None) or (s['anchor_menu'] in final_menus)
            error = None
        except Exception as e:
            elapsed = time.perf_counter() - t_start
            ok = False; all5_pass = False; unreal = None; anchor_preserved = None
            flags = {n: None for n in NUT_ORDER}
            final_menus = None
            error = f'{type(e).__name__}: {e}'
        all_run_rows.append({
            'repeat': rep, 'anchor_type': s['anchor_type'], 'anchor_menu': s['anchor_menu'],
            'scenario_idx': s['scenario_idx'], 'success_ok': ok, 'all5_nutrient_pass': all5_pass,
            'unrealistic': unreal, 'anchor_preserved': anchor_preserved,
            '열량_pass': flags['열량'], '단백질_pass': flags['단백질'], '칼륨_pass': flags['칼륨'],
            '인_pass': flags['인'], '나트륨_pass': flags['나트륨'],
            'elapsed_ms': elapsed * 1000, 'error': error, 'final_menus': '|'.join(final_menus) if final_menus else None,
        })
    n_err = sum(1 for r in all_run_rows if r['repeat'] == rep and r['error'])
    print(f'  완료 (에러 {n_err}건)')

print(f'\n총 실행: {len(all_run_rows)}건({N_REPEATS}회 x 300시나리오), 소요 {time.perf_counter() - t0:.1f}s')

runs_csv = os.path.join(OUT_DIR, 'final_service_core_metrics_runs_raw.csv')
with open(runs_csv, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(all_run_rows[0].keys()))
    w.writeheader(); w.writerows(all_run_rows)
print(f'저장(원시 900행): {runs_csv}')

# ============================================================
# 반복별 집계 -> 3회 평균±표준편차 (final_service_core_metrics.csv)
# ============================================================
import pandas as pd
df = pd.DataFrame(all_run_rows)

METRIC_DEFS = [
    ('all5_nutrient_pass_rate', 'all5_nutrient_pass', '5대 영양소 완전 충족률 (최종 선택 식단 기준)'),
    ('anchor_preserved_rate', 'anchor_preserved', '사용자 지정 메뉴 보존율 (두부콩류/생선구이/육류 225개 시나리오만, 랜덤 제외)'),
    ('unrealistic_rate', 'unrealistic', '비현실적 재료량 발생률'),
    ('success_rate', 'success_ok', '적합 식단 생성 성공률 (=1-fallback 발생률, 같은 정의라 하나만 보고)'),
    ('열량_pass_rate', '열량_pass', '열량 충족률'),
    ('단백질_pass_rate', '단백질_pass', '단백질 충족률'),
    ('칼륨_pass_rate', '칼륨_pass', '칼륨 충족률'),
    ('인_pass_rate', '인_pass', '인(raw P) 충족률'),
    ('나트륨_pass_rate', '나트륨_pass', '나트륨 충족률'),
]

per_repeat_rows = []
for rep in range(N_REPEATS):
    sub = df[df['repeat'] == rep]
    sub_nonrandom = sub[sub['anchor_type'] != '랜덤']
    row = {'repeat': rep, 'n_scenarios': len(sub)}
    for key, col, _ in METRIC_DEFS:
        if key == 'anchor_preserved_rate':
            row[key] = sub_nonrandom[col].mean()
        else:
            row[key] = sub[col].mean()
    per_repeat_rows.append(row)
per_repeat_df = pd.DataFrame(per_repeat_rows)
print('\n=== 반복별 지표 ===')
print(per_repeat_df.to_string(index=False))

summary_rows = []
for key, col, desc in METRIC_DEFS:
    vals = per_repeat_df[key].tolist()
    summary_rows.append({
        'metric': key, 'description': desc,
        'mean': statistics.mean(vals), 'std': statistics.stdev(vals) if len(vals) > 1 else 0.0,
        'repeat1': vals[0], 'repeat2': vals[1], 'repeat3': vals[2],
    })
summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUT_DIR, 'final_service_core_metrics.csv')
summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
print(f'\n저장: {summary_csv}')
print(summary_df.to_string(index=False))

# 앵커별 분해(주요 실패 원인 파악용, 3회 평균)
anchor_rows = []
for a in ['두부콩류', '생선구이', '육류', '랜덤']:
    per_rep_vals = {k: [] for k, _, _ in METRIC_DEFS}
    for rep in range(N_REPEATS):
        sub = df[(df['repeat'] == rep) & (df['anchor_type'] == a)]
        for key, col, _ in METRIC_DEFS:
            if key == 'anchor_preserved_rate' and a == '랜덤':
                continue
            per_rep_vals[key].append(sub[col].mean())
    row = {'anchor_type': a}
    for key, vals in per_rep_vals.items():
        if vals:
            row[f'{key}_mean'] = statistics.mean(vals)
    anchor_rows.append(row)
anchor_df = pd.DataFrame(anchor_rows)
anchor_csv = os.path.join(OUT_DIR, 'final_service_core_metrics_by_anchor.csv')
anchor_df.to_csv(anchor_csv, index=False, encoding='utf-8-sig')
print(f'\n저장(부속, 앵커별): {anchor_csv}')
print(anchor_df.to_string(index=False))

# 주요 실패 원인 샘플(전체 900건 중 all5_nutrient_pass=False인 사례의 실패 사유 분포)
fail_df = df[df['all5_nutrient_pass'] == False]
fail_reason_counts = {n: int((~fail_df[f'{n}_pass'].astype(bool)).sum()) for n in NUT_ORDER}
print(f'\n=== 실패({len(fail_df)}건) 중 영양소별 미충족 건수(중복가능, 여러 영양소 동시미달 가능) ===')
print(fail_reason_counts)
with open(os.path.join(OUT_DIR, '_fail_reason_counts.txt'), 'w', encoding='utf-8') as f:
    f.write(str(fail_reason_counts))
