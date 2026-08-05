# -*- coding: utf-8 -*-
"""
final_service_benchmark_120_realistic_weight_FOOK.py — 이전 120건 평가(final_service_
benchmark_120_FOOK.py)의 "표준체중 90kg+" 시나리오가 비현실적(그런 표준체중이 나오려면
키 202~207cm 필요)이었음을 사용자가 지적해 정정한 재평가.

이번엔 체중을 직접 넣지 않는다. 현실적 성인 키(145~190cm)를 먼저 만들고, 서비스가 실제
쓰는 공식 F.standard_weight(height_cm, sex) = 키(m)^2 x (남22/여21)로 표준체중을 계산해서만
사용한다. 그 외 120개 시나리오 구조(tier 60/40/20, 요청방식, 누적영양상태, 메뉴·재료 풀,
meals_left)는 이전 스크립트와 최대한 동일하게 유지한다.

서비스 코드/체크포인트/레버 코드는 수정하지 않는다. 평가·보고만 수행.
"""
import os, sys, csv, time
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

OUT_DIR = os.path.join(CODE, 'final_service_benchmark_out')
os.makedirs(OUT_DIR, exist_ok=True)

print('app_core_FOOK 임포트 (production: RL i002 + 현재 최종 레버)...')
import app_core_FOOK as core   # noqa: E402
print(f'  core.CKPT = {core.CKPT}')
print(f'  최종 레버 함수 존재(있어야 함): {hasattr(core.F, "lever_phosphorus_rawP")}')
print(f'  표준체중 공식 실제 사용 확인: F.standard_weight(170,"남")={core.F.standard_weight(170,"남"):.2f}kg'
      f' (기대값 22*1.7^2=63.58)')

NUT_ORDER = ['열량', '단백질', '칼륨', '인', '나트륨']
CONSUMED_KEYS = ('E', 'protein', 'K', 'P', 'Na', 'Na_season')


def nutrient_flags(t, bb):
    return {'열량': bb['Elo'] <= t['E'] <= bb['Ehi'],
            '단백질': bb['Plo'] <= t['protein'] <= bb['Phi'],
            '칼륨': t['K'] < bb['Kmax'],
            '인': t['P'] < bb['Pmax'],
            '나트륨': t['Na_season'] <= bb['Namax']}


PRIOR_ANCHORS = {'두부양념조림', '고등어구이', '제육불고기',
                 '삼치구이', '방어구이', '소불고기', '닭볶음탕', '오징어볶음', '새우튀김',
                 '오리불고기', '낙지볶음', '스크램블에그', '우엉조림', '마파두부', '함박스테이크'}
PRIOR_ING_KEYWORDS = {'곤드레', '고비', '돌나물', '머위', '리코타', '곶감', '녹두', '딸기',
                       '바지락', '고춧잎', '취나물', '갯기름나물'}

NEW_MAIN_POOL = [
    '가지튀김양념장', '감자고추장조림', '고추잡채꽃빵', '김달걀말이', '너비아니구이',
    '늙은호박조림', '닭갈비', '돈갈비찜', '동태무조림', '두부구이', '떡갈비구이',
    '맛살브로콜리볶음', '미트볼조림', '브로콜리볶음', '비엔나소시지볶음', '삼겹살두루치기',
    '아귀찜', '애호박전', '연근조림', '옥수수전', '임연수구이', '한식잡채', '치즈달걀말이',
    '코다리강정', '표고버섯볶음', '해물파전', '훈제오리구이', '닭강정',
]
NEW_MAIN_POOL = [m for m in NEW_MAIN_POOL if m not in PRIOR_ANCHORS]

NEW_ING_KEYWORD_POOL = [
    '꽃게', '날치알', '동치미', '검은콩두유', '오렌지', '귤', '깍두기', '무지개떡', '메론',
    '바나나', '배', '복숭아', '백설기', '결명자차', '두유', '녹차', '햄', '매실', '멜론',
    '땅콩샌드', '크래커',
]
NEW_ING_KEYWORD_POOL = [k for k in NEW_ING_KEYWORD_POOL if k not in PRIOR_ING_KEYWORDS]

print('\n신규 재료 키워드 사전검증 (make_meal(ingredient=) 실제 매칭 확인)...')
np.random.seed(1)
verified_ings = []
for kw in NEW_ING_KEYWORD_POOL:
    cand, note, b, anchor, warn = core.make_meal(ingredient=kw, W=60)
    if anchor is not None and anchor not in PRIOR_ANCHORS:
        verified_ings.append((kw, str(anchor)))
seen_menus = set(); dedup_ings = []
for kw, m in verified_ings:
    if m not in seen_menus:
        dedup_ings.append((kw, m)); seen_menus.add(m)
print(f'  검증+중복제거 후: {len(dedup_ings)}개')

# ============================================================
# 현실적 키(145~190cm) x 성별 -> 서비스 실제 공식으로 표준체중 계산
# ============================================================
HEIGHT_BANDS = {
    'H145-157': [146, 149, 152, 155, 157],
    'H158-167': [159, 161, 163, 165, 167],
    'H168-179': [169, 172, 175, 177, 179],
    'H180-190': [181, 183, 185, 187, 190],
}
SEX_CYCLE = ['남', '여']


def height_sex_weight(hb, k, sex_offset=0):
    h = HEIGHT_BANDS[hb][k % len(HEIGHT_BANDS[hb])]
    sex = SEX_CYCLE[(k + sex_offset) % 2]
    w = round(core.F.standard_weight(h, sex), 1)
    return h, sex, w


main_cycle = list(NEW_MAIN_POOL)
ing_cycle = list(dedup_ings)


def next_main(i):
    return main_cycle[i % len(main_cycle)]


def next_ing(i):
    return ing_cycle[i % len(ing_cycle)]


scenarios = []
sid_counter = 0


def add_scenario(tier, height, sex, weight, mode, cum_state, meals_left, anchor_or_ing_input, desc):
    global sid_counter
    sid_counter += 1
    scenarios.append({
        'sid': f'R{sid_counter:03d}', 'tier': tier, 'height': height, 'sex': sex, 'weight': weight,
        'mode': mode, 'cum_state': cum_state, 'meals_left': meals_left,
        'day_context': meals_left < 3,
        'anchor_or_ing_input': anchor_or_ing_input, 'desc': desc,
    })


hb_names = list(HEIGHT_BANDS.keys())

# --- 일반 조건 60개: 4키구간 x 3모드 x 5개(=60) ---
mi = 0; ii = 0
idx = 0
for hb in hb_names:
    for mode in ['random', 'menu', 'ingredient']:
        for k in range(5):
            h, sex, w = height_sex_weight(hb, k, sex_offset=hb_names.index(hb))
            if mode == 'random':
                target = None; ml = 3 if k % 2 == 0 else 2; cs = '여유' if ml == 3 else '보통'
            elif mode == 'menu':
                target = next_main(mi); mi += 1; ml = 3 if k % 2 == 0 else 2; cs = '여유' if ml == 3 else '보통'
            else:
                target = next_ing(ii)[0]; ii += 1; ml = 3 if k % 2 == 0 else 2; cs = '여유' if ml == 3 else '보통'
            add_scenario('1_일반', h, sex, w, mode, cs, ml, target,
                         f'일반조건: 키{h}cm/{sex}(표준체중{w}kg,{hb}), {mode}모드, meals_left={ml}({cs})')
            idx += 1
assert idx == 60, idx

# --- 난도높은 조건 40개: 고신장(표준체중최고) + 랜덤/메뉴 20 + 중간신장+빡빡한누적 10 + 저신장극단 10 ---
n_hard = 0
for k in range(10):
    h, sex, w = height_sex_weight('H180-190', k, sex_offset=0)
    add_scenario('2_난도높음', h, sex, w, 'random', '보통', 3, None,
                 f'난도높음: 고신장{h}cm/{sex}(표준체중{w}kg, 최고구간) + 랜덤모드(취약조합, 첫끼)')
    n_hard += 1
for k in range(10):
    h, sex, w = height_sex_weight('H180-190', k, sex_offset=1)
    m = next_main(mi); mi += 1
    add_scenario('2_난도높음', h, sex, w, 'menu', '보통', 3, m,
                 f'난도높음: 고신장{h}cm/{sex}(표준체중{w}kg) + 메뉴지정="{m}"(신규앵커, 첫끼)')
    n_hard += 1
for k in range(10):
    hb = ['H158-167', 'H168-179'][k % 2]
    h, sex, w = height_sex_weight(hb, k, sex_offset=k)
    mode = ['random', 'menu', 'ingredient'][k % 3]
    if mode == 'menu':
        target = next_main(mi); mi += 1
    elif mode == 'ingredient':
        target = next_ing(ii)[0]; ii += 1
    else:
        target = None
    add_scenario('2_난도높음', h, sex, w, mode, '빡빡함', 2, target,
                 f'난도높음: 키{h}cm/{sex}(표준체중{w}kg) + meals_left=2(1끼 소비 후, 누적예산 빡빡) + {mode}모드')
    n_hard += 1
for k in range(10):
    h, sex, w = height_sex_weight('H145-157', k, sex_offset=k)
    mode = ['random', 'ingredient'][k % 2]
    if mode == 'ingredient':
        target = next_ing(ii)[0]; ii += 1
    else:
        target = None
    add_scenario('2_난도높음', h, sex, w, mode, '보통', 3, target,
                 f'난도높음: 저신장{h}cm/{sex}(표준체중{w}kg, 극단) + {mode}모드')
    n_hard += 1
assert n_hard == 40, n_hard

# --- 경계조건 20개: meals_left=1(마지막끼) + 저/고신장 극단 + 메뉴지정 위주 ---
n_bound = 0
for k in range(20):
    hb = 'H145-157' if k < 10 else 'H180-190'
    h, sex, w = height_sex_weight(hb, k, sex_offset=k)
    if k % 3 == 0:
        mode = 'menu'; target = next_main(mi); mi += 1
    elif k % 3 == 1:
        mode = 'ingredient'; target = next_ing(ii)[0]; ii += 1
    else:
        mode = 'random'; target = None
    add_scenario('3_경계조건', h, sex, w, mode, '매우빡빡함', 1, target,
                 f'경계조건: 키{h}cm/{sex}(표준체중{w}kg) + meals_left=1(마지막끼, 2끼 소비 후) + {mode}모드')
    n_bound += 1
assert n_bound == 20, n_bound

assert len(scenarios) == 120, len(scenarios)
print(f'\n총 시나리오: {len(scenarios)}개 (일반60/난도40/경계20) 구성 완료')

overlap = [s for s in scenarios if s['anchor_or_ing_input'] in PRIOR_ANCHORS]
assert not overlap, f'이전 평가와 겹치는 앵커 발견: {overlap}'
print('이전 300/60/120건 시나리오와의 앵커/재료 중복 없음 확인.')

heights_used = [s['height'] for s in scenarios]
weights_used = [s['weight'] for s in scenarios]
print(f'실제 사용한 키 범위: {min(heights_used)}~{max(heights_used)}cm')
print(f'계산된 표준체중 범위: {min(weights_used):.1f}~{max(weights_used):.1f}kg')

scen_csv = os.path.join(OUT_DIR, 'final_service_benchmark_120_realistic_weight.csv')
with open(scen_csv, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(scenarios[0].keys()))
    w.writeheader(); w.writerows(scenarios)
print(f'저장(고정, 이후 절대 수정하지 않음): {scen_csv}')

import collections
print('\n=== 구성 분포 확인 ===')
print('tier:', collections.Counter(s['tier'] for s in scenarios))
print('mode:', collections.Counter(s['mode'] for s in scenarios))
print('sex:', collections.Counter(s['sex'] for s in scenarios))
print('cum_state:', collections.Counter(s['cum_state'] for s in scenarios))
print('meals_left:', collections.Counter(s['meals_left'] for s in scenarios))


def weight_band_of(w):
    if w < 55:
        return 'SW44-54'
    elif w < 65:
        return 'SW55-64'
    elif w < 75:
        return 'SW65-74'
    else:
        return 'SW75-80'


def height_band_of(h):
    for hb, hs in HEIGHT_BANDS.items():
        lo, hi = hb.replace('H', '').split('-')
        if int(lo) <= h <= int(hi):
            return hb
    return 'other'


wb_count = collections.Counter(weight_band_of(s['weight']) for s in scenarios)
hb_count = collections.Counter(height_band_of(s['height']) for s in scenarios)
print('weight_band(표준체중):', wb_count)
print('height_band(키):', hb_count)


def run_one(weight, mode, meals_left, target, day_context):
    error = None
    prior_consumed = None
    try:
        if day_context:
            consumed = {k: 0 for k in CONSUMED_KEYS}
            for mi_ in range(3 - meals_left):
                bounds_i = core.F.meal_bounds(weight, consumed, meals_left=3 - mi_)
                cand_i, _, _, _, _ = core.make_meal(menu=None, W=weight, bounds=bounds_i)
                _, _, after_i, _ = cand_i
                for k in CONSUMED_KEYS:
                    consumed[k] = round(consumed[k] + after_i[k], 4)
            prior_consumed = dict(consumed)
            bounds_test = core.F.meal_bounds(weight, consumed, meals_left=meals_left)
        else:
            bounds_test = core.F.meal_bounds(weight)

        menu_arg = target if mode == 'menu' else None
        ing_arg = target if mode == 'ingredient' else None
        t_start = time.perf_counter()
        cand, note, b, resolved_anchor, warn = core.make_meal(menu=menu_arg, ingredient=ing_arg,
                                                                W=weight, bounds=bounds_test)
        elapsed = time.perf_counter() - t_start
        menus, inst, after, ok = cand
        flags = nutrient_flags(after, b)
        all5_pass = all(flags.values())
        unreal = core.F.unrealistic_reason(inst)
        final_menus = list(dict.fromkeys(i['menu'] for i in inst))
        target_menu = resolved_anchor if mode == 'ingredient' else menu_arg
        anchor_preserved = (target_menu is None) or (target_menu in final_menus)
        return {
            'resolved_anchor': resolved_anchor, 'prior_consumed': prior_consumed,
            'success_ok': ok, 'all5_nutrient_pass': all5_pass,
            'unrealistic': unreal is not None, 'unreal_reason': unreal,
            'anchor_preserved': anchor_preserved,
            '열량_pass': flags['열량'], '단백질_pass': flags['단백질'], '칼륨_pass': flags['칼륨'],
            '인_pass': flags['인'], '나트륨_pass': flags['나트륨'],
            'final_menus': '|'.join(final_menus), 'elapsed_ms': elapsed * 1000, 'error': None,
        }
    except Exception as e:
        return {
            'resolved_anchor': None, 'prior_consumed': prior_consumed,
            'success_ok': None, 'all5_nutrient_pass': None, 'unrealistic': None, 'unreal_reason': None,
            'anchor_preserved': None, '열량_pass': None, '단백질_pass': None, '칼륨_pass': None,
            '인_pass': None, '나트륨_pass': None, 'final_menus': None, 'elapsed_ms': None,
            'error': f'{type(e).__name__}: {e}',
        }


print('\n' + '=' * 70)
print('120개 시나리오 실행 (각 1회, tries=48)')
print('=' * 70)
bench_results = []
t0 = time.perf_counter()
for i, s in enumerate(scenarios):
    seed_val = 810000 + i * 977
    np.random.seed(seed_val)
    r = run_one(s['weight'], s['mode'], s['meals_left'], s['anchor_or_ing_input'], s['day_context'])
    row = dict(s); row.update(r)
    row['weight_band'] = weight_band_of(s['weight']); row['height_band'] = height_band_of(s['height'])
    row['repeat'] = 0
    bench_results.append(row)
    if (i + 1) % 20 == 0:
        print(f'  {i+1}/120 완료')
print(f'120건 실행 완료, 소요 {time.perf_counter()-t0:.1f}s')

bench_csv = os.path.join(OUT_DIR, 'final_service_benchmark_results_realistic_weight.csv')
fieldnames = list(bench_results[0].keys())
with open(bench_csv, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader(); w.writerows(bench_results)
print(f'저장: {bench_csv}')

n_fail = sum(1 for r in bench_results if r['success_ok'] is False)
n_err = sum(1 for r in bench_results if r['error'])
print(f'전체 성공률: {sum(1 for r in bench_results if r["success_ok"])}/120, 실패 {n_fail}건, 오류 {n_err}건')


def pick_representative(tier_name, n_pick, all_rows):
    rows = [r for r in all_rows if r['tier'] == tier_name]
    fails = [r for r in rows if r['success_ok'] is False]
    passes = [r for r in rows if r['success_ok'] is True]
    n_fail_pick = (n_pick + 1) // 2
    n_pass_pick = n_pick - n_fail_pick

    def diverse_pick(pool, n):
        picked = []; seen_wb = set()
        for r in pool:
            if len(picked) >= n:
                break
            if r['weight_band'] not in seen_wb:
                picked.append(r); seen_wb.add(r['weight_band'])
        for r in pool:
            if len(picked) >= n:
                break
            if r not in picked:
                picked.append(r)
        return picked[:n]

    picked_fail = diverse_pick(fails, min(n_fail_pick, len(fails)))
    picked_pass = diverse_pick(passes, min(n_pass_pick, len(passes)))
    picked = picked_fail + picked_pass
    remaining = [r for r in rows if r not in picked]
    while len(picked) < n_pick and remaining:
        picked.append(remaining.pop(0))
    return picked[:n_pick]


rep20 = (pick_representative('1_일반', 10, bench_results) +
         pick_representative('2_난도높음', 7, bench_results) +
         pick_representative('3_경계조건', 3, bench_results))
assert len(rep20) == 20, len(rep20)
print(f'\n대표 20개 선정 완료: {[r["sid"] for r in rep20]}')
print(f'  선정건 중 최초실행 성공: {sum(1 for r in rep20 if r["success_ok"])}, 실패: {sum(1 for r in rep20 if r["success_ok"] is False)}')

print('\n' + '=' * 70)
print('대표 20개 반복안정성 확인 (서로 다른 시드로 2회 추가 실행, 총 40건)')
print('=' * 70)
repeat_results = []
for r in rep20:
    row = dict(r); row['repeat'] = 1
    repeat_results.append(row)

for rep in (2, 3):
    for r in rep20:
        seed_val = 910000 + rep * 500000 + int(r['sid'][1:]) * 977
        np.random.seed(seed_val)
        res = run_one(r['weight'], r['mode'], r['meals_left'], r['anchor_or_ing_input'], r['day_context'])
        row = {k: r[k] for k in ['sid', 'tier', 'height', 'sex', 'weight', 'mode', 'cum_state', 'meals_left',
                                   'day_context', 'anchor_or_ing_input', 'desc', 'weight_band', 'height_band']}
        row.update(res); row['repeat'] = rep
        repeat_results.append(row)
print(f'  반복실행 완료: {len(repeat_results)}건')

repeat_csv = os.path.join(OUT_DIR, 'final_service_repeat_20_realistic_weight.csv')
fieldnames_r = list(repeat_results[0].keys())
with open(repeat_csv, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames_r)
    w.writeheader(); w.writerows(repeat_results)
print(f'저장: {repeat_csv}')

stability_rows = []
for sid in [r['sid'] for r in rep20]:
    sub = [r for r in repeat_results if r['sid'] == sid]
    sub.sort(key=lambda x: x['repeat'])
    successes = [r['success_ok'] for r in sub]
    menus_list = [r['final_menus'] for r in sub]
    changed_outcome = len(set(successes)) > 1
    changed_menu = len(set(menus_list)) > 1
    stability_rows.append({'sid': sid, 'tier': sub[0]['tier'], 'weight_band': sub[0]['weight_band'],
                            'success_repeat1': successes[0], 'success_repeat2': successes[1],
                            'success_repeat3': successes[2], 'outcome_changed': changed_outcome,
                            'menu_changed_across_repeats': changed_menu})
print('\n=== 반복안정성 요약 (20개) ===')
n_outcome_changed = sum(1 for r in stability_rows if r['outcome_changed'])
n_menu_changed = sum(1 for r in stability_rows if r['menu_changed_across_repeats'])
print(f'  성공/실패가 반복간 바뀐 시나리오: {n_outcome_changed}/20')
print(f'  최종메뉴 조합이 반복간 달랐던 시나리오: {n_menu_changed}/20')
for r in stability_rows:
    print(f"  {r['sid']}({r['tier']}/{r['weight_band']}): {r['success_repeat1']},{r['success_repeat2']},{r['success_repeat3']} "
          f"outcome_changed={r['outcome_changed']}")

print(f'\n총 실행 횟수: 120(1차) + 40(추가 반복) = {120+40}회')