# -*- coding: utf-8 -*-
"""
external_condition_test_FOOK.py — 최종 production 서비스(RL i002 + 현재 최종
FOOK_adjust_levers.py)의 일반화 가능성 확인용 소규모 외부조건 테스트.

기존 300개 내부평가 시나리오(체중60kg 고정, 대표앵커 3종+랜덤)와 겹치지 않는 신규 60개
시나리오(5개 유형 x 12개)를 app_core_FOOK.make_meal() 직접호출(tries=48, 실제 서비스
기본값)로 1회씩만 평가한다. 서비스 코드/체크포인트/레버 코드는 수정하지 않는다.

5개 유형:
  1. 체중변화   : 저/중/고 체중 12개(45~92kg, 60kg 미포함), 랜덤모드
  2. 누적영양상태: day_result()와 동일한 3끼 누적 로직을 직접 재현(consumed 조작 없이 실제
                  1·2끼 생성 결과를 누적) - 3번째 끼(meals_left=1)를 시나리오로 채점
  3. 신규앵커   : 두부양념조림/고등어구이/제육불고기 외 12개 실제 메뉴(다른 식품군)
  4. 희귀재료   : 실제 menu_ings에서 1~2개 메뉴에만 등장하는 재료 12개 (ingredient= 모드,
                  사전에 실제 make_meal(ingredient=...) 호출로 매칭 확인된 것만 사용)
  5. 경계조건   : 저체중 + meals_left=1(3번째 끼) + 가장 어려운 앵커(두부양념조림) 결합
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

OUT_DIR = os.path.join(CODE, 'external_condition_test_out')
os.makedirs(OUT_DIR, exist_ok=True)

print('app_core_FOOK 임포트 (production: RL i002 + 현재 최종 레버)...')
import app_core_FOOK as core   # noqa: E402
print(f'  core.CKPT = {core.CKPT}')
print(f'  최종 레버 함수 존재(있어야 함): {hasattr(core.F, "lever_phosphorus_rawP")}')

NUT_ORDER = ['열량', '단백질', '칼륨', '인', '나트륨']


def nutrient_flags(t, bb):
    return {'열량': bb['Elo'] <= t['E'] <= bb['Ehi'],
            '단백질': bb['Plo'] <= t['protein'] <= bb['Phi'],
            '칼륨': t['K'] < bb['Kmax'],
            '인': t['P'] < bb['Pmax'],
            '나트륨': t['Na_season'] <= bb['Namax']}


# ============================================================
# 60개 시나리오 정의
# ============================================================
scenarios = []

# --- 유형1: 체중변화 (12개, 60kg 제외, 저/중/고 분포) ---
WEIGHTS_1 = [45, 48, 50, 55, 58, 63, 65, 70, 75, 80, 85, 92]
for i, w in enumerate(WEIGHTS_1):
    scenarios.append({'sid': f'W{i+1:02d}', 'category': '1_체중변화', 'weight': w,
                       'mode': 'random', 'anchor_or_ing': None, 'day_context': False, 'meals_left': 3,
                       'desc': f'체중{w}kg, 랜덤모드, 첫끼(예산 전체)'})

# --- 유형2: 서로 다른 누적 영양 상태 (12개, day-context 3번째 끼) ---
WEIGHTS_2 = [50, 53, 56, 59, 62, 64, 66, 68, 72, 76, 80, 85]
for i, w in enumerate(WEIGHTS_2):
    scenarios.append({'sid': f'C{i+1:02d}', 'category': '2_누적영양상태', 'weight': w,
                       'mode': 'random', 'anchor_or_ing': None, 'day_context': True, 'meals_left': 1,
                       'desc': f'체중{w}kg, 실제1·2끼 생성 후 누적, 3번째끼(meals_left=1) 랜덤모드'})

# --- 유형3: 신규 대표앵커 (12개, 기존 3종 제외, 실제 메뉴·식품군 다양화) ---
NEW_ANCHORS = ['삼치구이', '방어구이', '소불고기', '닭볶음탕', '오징어볶음', '새우튀김',
               '오리불고기', '낙지볶음', '스크램블에그', '우엉조림', '마파두부', '함박스테이크']
for i, m in enumerate(NEW_ANCHORS):
    scenarios.append({'sid': f'A{i+1:02d}', 'category': '3_신규앵커', 'weight': 60,
                       'mode': 'menu', 'anchor_or_ing': m, 'day_context': False, 'meals_left': 3,
                       'desc': f'체중60kg, 메뉴지정="{m}"(기존 3앵커 외 신규, 첫끼)'})

# --- 유형4: 희귀 재료 요청 (12개, 실제 make_meal(ingredient=)로 매칭 사전확인됨) ---
RARE_INGS = ['곤드레', '고비', '돌나물', '머위', '리코타', '곶감', '녹두', '딸기', '바지락',
             '고춧잎', '취나물', '갯기름나물']
for i, ing in enumerate(RARE_INGS):
    scenarios.append({'sid': f'I{i+1:02d}', 'category': '4_희귀재료', 'weight': 60,
                       'mode': 'ingredient', 'anchor_or_ing': ing, 'day_context': False, 'meals_left': 3,
                       'desc': f'체중60kg, 재료지정="{ing}"(학습데이터 내 1~2개 메뉴에만 등장하는 희귀재료, 첫끼)'})

# --- 유형5: 경계조건 (저체중 + meals_left=1 + 두부양념조림 앵커 결합) ---
WEIGHTS_5 = [45, 47, 49, 50, 52, 54, 56, 58, 61, 64, 67, 70]
for i, w in enumerate(WEIGHTS_5):
    scenarios.append({'sid': f'B{i+1:02d}', 'category': '5_경계조건', 'weight': w,
                       'mode': 'menu', 'anchor_or_ing': '두부양념조림', 'day_context': True, 'meals_left': 1,
                       'desc': f'체중{w}kg(저~중체중), 실제1·2끼 생성 후 누적, 3번째끼(meals_left=1)+두부양념조림 지정(가장 어려운 앵커)'})

assert len(scenarios) == 60, len(scenarios)
print(f'\n총 시나리오: {len(scenarios)}개 구성 완료')

scen_csv = os.path.join(OUT_DIR, 'external_condition_test_scenarios.csv')
with open(scen_csv, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(scenarios[0].keys()))
    w.writeheader(); w.writerows(scenarios)
print(f'저장: {scen_csv}')


# ============================================================
# 60개 시나리오 실행 (각 1회) — app_core_FOOK.make_meal() 직접호출, tries=48(기본값)
# day_context=True인 시나리오는 day_result()와 동일한 누적로직(meals_left 3->2->1)을
# 실제 make_meal() 2회(1·2번째 끼, 랜덤모드)로 먼저 실행해 consumed를 만든 뒤,
# 3번째 끼(시나리오 본체)를 meals_left=1로 평가한다 - 선택로직 재구현이 아니라
# day_result()가 하는 '누적 호출'을 그대로 반복한 것뿐, make_meal() 자체는 매번 실제 호출.
# ============================================================
CONSUMED_KEYS = ('E', 'protein', 'K', 'P', 'Na', 'Na_season')
results = []
t0 = time.perf_counter()
for idx, s in enumerate(scenarios):
    # 주의: Python 내장 hash()는 문자열에 대해 프로세스마다 랜덤화(PYTHONHASHSEED)되어
    # 재실행 시 값이 달라진다 - 재현성 확보를 위해 순수 정수 연산만으로 시드를 만든다.
    seed_val = 700000 + idx * 977
    np.random.seed(seed_val)
    error = None
    prior_consumed = None
    try:
        if s['day_context']:
            consumed = {k: 0 for k in CONSUMED_KEYS}
            for mi in range(2):   # 1·2번째 끼(랜덤모드)로 실제 누적 생성
                bounds_i = core.F.meal_bounds(s['weight'], consumed, meals_left=3 - mi)
                cand_i, _, _, _, _ = core.make_meal(menu=None, W=s['weight'], bounds=bounds_i)
                _, _, after_i, _ = cand_i
                for k in CONSUMED_KEYS:
                    consumed[k] = round(consumed[k] + after_i[k], 4)
            prior_consumed = dict(consumed)
            bounds_test = core.F.meal_bounds(s['weight'], consumed, meals_left=s['meals_left'])
        else:
            bounds_test = core.F.meal_bounds(s['weight'])

        menu_arg = s['anchor_or_ing'] if s['mode'] == 'menu' else None
        ing_arg = s['anchor_or_ing'] if s['mode'] == 'ingredient' else None
        t_start = time.perf_counter()
        cand, note, b, resolved_anchor, warn = core.make_meal(menu=menu_arg, ingredient=ing_arg,
                                                                W=s['weight'], bounds=bounds_test)
        elapsed = time.perf_counter() - t_start
        menus, inst, after, ok = cand
        flags = nutrient_flags(after, b)
        all5_pass = all(flags.values())
        unreal = core.F.unrealistic_reason(inst)
        final_menus = list(dict.fromkeys(i['menu'] for i in inst))
        if s['mode'] == 'ingredient':
            target_menu = resolved_anchor   # 재료검색이 실제로 골라준 메뉴명(사전확인된 값과 일치해야 정상)
        else:
            target_menu = menu_arg
        anchor_preserved = (target_menu is None) or (target_menu in final_menus)
        after_snapshot = {k: after[k] for k in ('E', 'protein', 'K', 'P', 'Na_season')}
        bounds_snapshot = {k: b[k] for k in ('Elo', 'Ehi', 'Plo', 'Phi', 'Kmax', 'Pmax', 'Namax')}
    except Exception as e:
        elapsed = None
        cand = note = b = resolved_anchor = warn = None
        menus = inst = after = ok = None
        flags = {n: None for n in NUT_ORDER}
        all5_pass = None; unreal = None; final_menus = None; anchor_preserved = None
        target_menu = s['anchor_or_ing']
        after_snapshot = None; bounds_snapshot = None
        error = f'{type(e).__name__}: {e}'

    results.append({
        'sid': s['sid'], 'category': s['category'], 'weight': s['weight'], 'mode': s['mode'],
        'anchor_or_ing_input': s['anchor_or_ing'], 'resolved_anchor': resolved_anchor,
        'meals_left': s['meals_left'], 'prior_consumed': prior_consumed,
        'success_ok': ok, 'all5_nutrient_pass': all5_pass,
        'unrealistic': (unreal is not None) if error is None else None,
        'unreal_reason': unreal, 'anchor_preserved': anchor_preserved,
        '열량_pass': flags['열량'], '단백질_pass': flags['단백질'], '칼륨_pass': flags['칼륨'],
        '인_pass': flags['인'], '나트륨_pass': flags['나트륨'],
        'final_menus': '|'.join(final_menus) if final_menus else None,
        'elapsed_ms': (elapsed * 1000) if elapsed is not None else None,
        'error': error, 'desc': s['desc'],
        'after_E': after_snapshot['E'] if after_snapshot else None,
        'after_protein': after_snapshot['protein'] if after_snapshot else None,
        'after_K': after_snapshot['K'] if after_snapshot else None,
        'after_P': after_snapshot['P'] if after_snapshot else None,
        'after_Na_season': after_snapshot['Na_season'] if after_snapshot else None,
        'b_Elo': bounds_snapshot['Elo'] if bounds_snapshot else None,
        'b_Ehi': bounds_snapshot['Ehi'] if bounds_snapshot else None,
        'b_Plo': bounds_snapshot['Plo'] if bounds_snapshot else None,
        'b_Phi': bounds_snapshot['Phi'] if bounds_snapshot else None,
        'b_Kmax': bounds_snapshot['Kmax'] if bounds_snapshot else None,
        'b_Pmax': bounds_snapshot['Pmax'] if bounds_snapshot else None,
        'b_Namax': bounds_snapshot['Namax'] if bounds_snapshot else None,
    })
    print(f"  [{idx+1:2d}/60] {s['sid']} ({s['category']}) -> "
          f"{'ERROR:' + error if error else ('ok=' + str(ok))}")

print(f'\n총 실행 완료: {len(results)}건, 소요 {time.perf_counter() - t0:.1f}s')

results_csv = os.path.join(OUT_DIR, 'external_condition_test_results.csv')
with open(results_csv, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader(); w.writerows(results)
print(f'저장: {results_csv}')

# ============================================================
# 핵심지표 집계 (전체 60건 + 카테고리별)
# ============================================================
n_total = len(results)
n_err = sum(1 for r in results if r['error'])
valid = [r for r in results if not r['error']]


def rate(rows, key, denom_filter=None):
    rr = [r for r in rows if denom_filter is None or denom_filter(r)]
    vals = [r[key] for r in rr if r[key] is not None]
    return (sum(1 for v in vals if v) / len(vals)) if vals else None


print('\n=== 전체 60건 핵심지표 ===')
print(f"  적합 식단 성공률(=1-fallback): {rate(valid, 'success_ok')}")
print(f"  5대영양 완전충족률: {rate(valid, 'all5_nutrient_pass')}")
print(f"  사용자지정메뉴 보존율(anchor 있는 건만): {rate(valid, 'anchor_preserved', lambda r: r['anchor_or_ing_input'] is not None)}")
print(f"  비현실재료량 발생률: {rate(valid, 'unrealistic')}")
print(f"  오류·예외 발생률: {n_err}/{n_total} = {n_err/n_total:.4f}")
for n in NUT_ORDER:
    print(f"  {n} 충족률: {rate(valid, f'{n}_pass')}")

print('\n=== 카테고리별 ===')
for cat in ['1_체중변화', '2_누적영양상태', '3_신규앵커', '4_희귀재료', '5_경계조건']:
    rows = [r for r in results if r['category'] == cat]
    err = sum(1 for r in rows if r['error'])
    print(f"  [{cat}] n={len(rows)} 성공률={rate(rows,'success_ok')} 5영양={rate(rows,'all5_nutrient_pass')} "
          f"보존율={rate(rows,'anchor_preserved', lambda r: r['anchor_or_ing_input'] is not None)} "
          f"비현실={rate(rows,'unrealistic')} 오류={err}")

print('\n=== 실패/fallback 사례(있으면) ===')
fail_rows = [r for r in valid if r['success_ok'] is False]
for r in fail_rows:
    print(f"  {r['sid']} ({r['category']}) 지정={r['anchor_or_ing_input']} "
          f"미충족영양={[n for n in NUT_ORDER if r[f'{n}_pass'] is False]} unreal={r['unreal_reason']}")
if not fail_rows:
    print('  없음 (60건 전부 success_ok=True)')
err_rows = [r for r in results if r['error']]
for r in err_rows:
    print(f"  [오류] {r['sid']} ({r['category']}) 지정={r['anchor_or_ing_input']} error={r['error']}")
