"""
custom sodium(총나트륨 하루 상한)이 최종 적합성 판정(passes()/day_ok)까지 hard constraint로
이어지는지 확인한다 — 2026-08 나트륨 재설계 2차 수정.

app_core_FOOK.py는 TF 모델 로딩 비용이 있어(다른 test_*_route*.py와 같은 이유) passes()의
소스를 그대로 잘라내 실행한다(재구현이 아니라 원본 텍스트 그대로 exec) — 실제로 조건이
바뀌면 이 테스트가 깨진다. FOOK_adjust_levers는 가벼워서(openpyxl만 씀) day_targets()/
meal_bounds()는 직접 import해서 쓴다 — b['custom_sodium_active']/b['Na_total_target']가
실제 함수가 계산한 진짜 값이다(테스트가 값을 지어내지 않음).
"""
import ast
import re
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
CORE_SRC = (BACKEND_DIR / 'app_core_FOOK.py').read_text(encoding='utf-8')

import sys  # noqa: E402
sys.path.insert(0, str(BACKEND_DIR))
import FOOK_adjust_levers as F  # noqa: E402


def _extract_function_source(src: str, name: str) -> str:
    m = re.search(rf'^def {re.escape(name)}\(', src, re.MULTILINE)
    assert m, f'{name}() 정의를 찾지 못했습니다.'
    lines = src[m.start():].splitlines(keepends=True)
    out = [lines[0]]
    for line in lines[1:]:
        if line.strip() == '' or line[:1] in (' ', '\t'):
            out.append(line)
        else:
            break
    return ''.join(out)


PASSES_SRC = _extract_function_source(CORE_SRC, 'passes')
_ns = {}
exec(PASSES_SRC, _ns)
passes = _ns['passes']

DAY_RESULT_SRC = _extract_function_source(CORE_SRC, 'day_result')


def _t(E=600, protein=24, K=500, P=300, Na=400, Na_season=200):
    return {'E': E, 'protein': protein, 'K': K, 'P': P, 'Na': Na, 'Na_season': Na_season}


# ---------------------------------------------------------------------------
# passes() 소스 자체가 custom_sodium_active를 조건부로 반영하는지(배선 확인)
# ---------------------------------------------------------------------------
def test_passes_source_checks_custom_sodium_active_conditionally():
    assert "b.get('custom_sodium_active')" in PASSES_SRC
    assert "t['Na'] <= b['Na_total_target']" in PASSES_SRC
    # 기존 5개 조건(첨가염 포함)은 지워지지 않았다.
    assert "t['Na_season'] <= b['Namax']" in PASSES_SRC


def test_day_result_source_checks_custom_sodium_active_conditionally():
    assert "dt.get('custom_sodium_active')" in DAY_RESULT_SRC
    assert "consumed['Na'] <= dt['Na_total_max']" in DAY_RESULT_SRC
    assert "consumed['Na_season'] <= dt['Namax']" in DAY_RESULT_SRC  # 기존 조건 유지


# ---------------------------------------------------------------------------
# 1~3. custom 없음 — 기존 판정과 완전히 동일해야 한다
# ---------------------------------------------------------------------------
def test_no_custom_passes_matches_original_five_conditions():
    b = F.meal_bounds(60, consumed=None, meals_left=3)
    assert b['custom_sodium_active'] is False
    t_pass = _t(E=b['Elo'] + 1, protein=b['Plo'] + 1, K=b['Kmax'] - 1, P=b['Pmax'] - 1,
                Na=999999, Na_season=b['Namax'])          # 총나트륨은 일부러 말도 안 되게 크게
    # custom 없으면 총나트륨이 얼마든 통과에 영향 없어야 한다(기존 5조건만 봄).
    assert passes(t_pass, b) is True
    t_fail = dict(t_pass, Na_season=b['Namax'] + 1)         # 첨가염만 넘기면 여전히 실패
    assert passes(t_fail, b) is False


def test_no_custom_day_ok_matches_original_condition():
    # day_ok 계산식 자체를 그대로 실행해 검증한다(실제 day_result()와 같은 5+1조건 수식).
    dt = F.day_targets(60)
    assert dt['custom_sodium_active'] is False
    consumed = {'E': dt['Elo'], 'protein': dt['Plo'], 'K': 0, 'P': 0,
                'Na': 999999, 'Na_season': dt['Namax']}     # 총나트륨 무한대여도
    EPS = 1e-6
    day_ok = (dt['Elo'] - EPS <= consumed['E'] <= dt['Ehi'] + EPS
              and dt['Plo'] - EPS <= consumed['protein'] <= dt['Phi'] + EPS
              and consumed['K'] < dt['Kmax'] + EPS and consumed['P'] < dt['Pmax'] + EPS
              and consumed['Na_season'] <= dt['Namax'] + EPS)
    if dt.get('custom_sodium_active'):
        day_ok = day_ok and consumed['Na'] <= dt['Na_total_max'] + EPS
    assert day_ok is True   # custom 없으면 총나트륨은 안 봄


def test_no_custom_generation_bounds_unchanged():
    # "생성 결과가 바뀌지 않는다"는 make_meal()이 후보를 고르는 기준(b)이 override 없을 때
    # 이전과 수치까지 완전히 같다는 뜻이다 — meal_bounds()가 실제로 그런지 직접 확인한다.
    b = F.meal_bounds(70, consumed={'E': 100, 'protein': 5, 'K': 50, 'P': 30, 'Na': 60}, meals_left=2)
    assert b['Namax'] == F.SALT_MG
    # cap(dv,used) = min((dv-used)/n, fair*(1+tol)) — 남은예산이 밴드 상한(fair*1.2)을 넘으면 클램프됨.
    fair = (F.NA_TOTAL_MEAL * 3) / 3
    expected = min((F.NA_TOTAL_MEAL * 3 - 60) / 2, fair * 1.2)
    assert b['Na_total_target'] == pytest.approx(expected)
    assert b['custom_sodium_active'] is False


# ---------------------------------------------------------------------------
# 4~5. custom sodium 있음 — 한 끼 판정
# ---------------------------------------------------------------------------
def test_custom_sodium_meal_fails_when_na_season_ok_but_total_na_exceeds():
    b = F.meal_bounds(60, consumed=None, meals_left=3, overrides={'sodium': 900})   # 300/끼
    assert b['custom_sodium_active'] is True
    t = _t(E=b['Elo'] + 1, protein=b['Plo'] + 1, K=b['Kmax'] - 1, P=b['Pmax'] - 1,
           Na_season=b['Namax'], Na=b['Na_total_target'] + 1)   # 첨가염 OK, 총나트륨만 초과
    assert passes(t, b) is False


def test_custom_sodium_meal_passes_when_both_within_bounds():
    b = F.meal_bounds(60, consumed=None, meals_left=3, overrides={'sodium': 900})
    t = _t(E=b['Elo'] + 1, protein=b['Plo'] + 1, K=b['Kmax'] - 1, P=b['Pmax'] - 1,
           Na_season=b['Namax'], Na=b['Na_total_target'])
    assert passes(t, b) is True


# ---------------------------------------------------------------------------
# 6~7. custom sodium 있음 — 하루 판정
# ---------------------------------------------------------------------------
def _day_ok(dt, consumed):
    EPS = 1e-6
    ok = (dt['Elo'] - EPS <= consumed['E'] <= dt['Ehi'] + EPS
          and dt['Plo'] - EPS <= consumed['protein'] <= dt['Phi'] + EPS
          and consumed['K'] < dt['Kmax'] + EPS and consumed['P'] < dt['Pmax'] + EPS
          and consumed['Na_season'] <= dt['Namax'] + EPS)
    if dt.get('custom_sodium_active'):
        ok = ok and consumed['Na'] <= dt['Na_total_max'] + EPS
    return ok


def test_custom_sodium_day_fails_when_na_season_ok_but_total_na_exceeds():
    dt = F.day_targets(60, {'sodium': 1500})
    assert dt['custom_sodium_active'] is True
    consumed = {'E': dt['Elo'], 'protein': dt['Plo'], 'K': 0, 'P': 0,
                'Na_season': dt['Namax'], 'Na': dt['Na_total_max'] + 1}   # 예시의 1700 > 1500과 같은 상황
    assert _day_ok(dt, consumed) is False


def test_custom_sodium_day_passes_when_total_na_within_custom_max():
    dt = F.day_targets(60, {'sodium': 1500})
    consumed = {'E': dt['Elo'], 'protein': dt['Plo'], 'K': 0, 'P': 0,
                'Na_season': dt['Namax'], 'Na': dt['Na_total_max']}
    assert _day_ok(dt, consumed) is True


# ---------------------------------------------------------------------------
# 8. custom sodium reset(안 보냄) → 다시 기존 판정 방식으로 복귀
# ---------------------------------------------------------------------------
def test_resetting_custom_sodium_restores_original_judging():
    b_custom = F.meal_bounds(60, consumed=None, meals_left=3, overrides={'sodium': 900})
    b_reset = F.meal_bounds(60, consumed=None, meals_left=3, overrides=None)   # "기본값으로 되돌리기"
    assert b_custom['custom_sodium_active'] is True
    assert b_reset['custom_sodium_active'] is False
    # 총나트륨이 아무리 커도 reset 후엔 다시 안 본다.
    t = _t(E=b_reset['Elo'] + 1, protein=b_reset['Plo'] + 1, K=b_reset['Kmax'] - 1,
           P=b_reset['Pmax'] - 1, Na_season=b_reset['Namax'], Na=999999)
    assert passes(t, b_reset) is True


# ---------------------------------------------------------------------------
# 첨가염(Namax/SALT_MG/lever_sodium) 자체는 이번 수정으로 손대지 않았는지 재확인
# ---------------------------------------------------------------------------
def test_namax_and_salt_mg_untouched_by_sodium_hard_constraint_change():
    b0 = F.meal_bounds(60, consumed=None, meals_left=3)
    b1 = F.meal_bounds(60, consumed=None, meals_left=3, overrides={'sodium': 500})
    assert b0['Namax'] == b1['Namax'] == F.SALT_MG == 393
    lever_sodium_src = re.search(r'^def lever_sodium\(inst\):.*?(?=^def )',
                                  (BACKEND_DIR / 'FOOK_adjust_levers.py').read_text(encoding='utf-8'),
                                  re.MULTILINE | re.DOTALL)
    assert lever_sodium_src and 'custom' not in lever_sodium_src.group(0)
