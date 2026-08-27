"""
개인별 영양 기준 override(custom_targets) 기능을 확인한다.

- schemas.CustomTargets는 TF 없이 바로 import 가능(schemas.py 자체가 가벼움)해서 직접 검증한다.
- FOOK_adjust_levers.day_targets()/meal_bounds()도 TF 의존이 없어(openpyxl만 씀) 직접 import해서
  순수 계산 결과를 검증한다 — 특히 나트륨은 두 값(첨가염 Namax / 총나트륨 Na_total_max)이
  섞이지 않는지가 핵심이다(2026-08 나트륨 재설계).
- server_FOOK.py/app_core_FOOK.py는 TF 모델 로딩 비용이 있어(다른 test_*_route*.py와 같은 이유)
  소스 문자열 수준에서 배선만 확인한다.
"""
import re
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from schemas import CustomTargets, GenReq, DayReq, DayTargetsReq, ProfileReq  # noqa: E402
import FOOK_adjust_levers as F  # noqa: E402


# ---------------------------------------------------------------------------
# CustomTargets 스키마 검증
# ---------------------------------------------------------------------------
def test_custom_targets_all_none_by_default():
    ct = CustomTargets()
    assert ct.model_dump(exclude_none=True) == {}


def test_custom_targets_partial_override_only_sets_given_fields():
    ct = CustomTargets(potassium=2200)
    d = ct.model_dump(exclude_none=True)
    assert d == {'potassium': 2200.0}


def test_custom_targets_energy_protein_need_lo_le_hi():
    with pytest.raises(ValidationError):
        CustomTargets(energy=[2200, 1800])   # lo > hi


def test_custom_targets_energy_protein_need_positive():
    with pytest.raises(ValidationError):
        CustomTargets(energy=[0, 1800])
    with pytest.raises(ValidationError):
        CustomTargets(protein=[-10, 50])


def test_custom_targets_energy_protein_need_exactly_two_values():
    with pytest.raises(ValidationError):
        CustomTargets(energy=[1800])


def test_custom_targets_upper_bound_fields_reject_non_positive():
    with pytest.raises(ValidationError):
        CustomTargets(potassium=0)
    with pytest.raises(ValidationError):
        CustomTargets(sodium=-5)


def test_custom_targets_upper_bound_fields_reject_absurd_values():
    with pytest.raises(ValidationError):
        CustomTargets(phosphorus=999999)


def test_custom_targets_rejects_bool_disguised_as_number():
    # bool은 파이썬에서 int 서브타입이라({'potassium': true} -> 1.0) 조용히 통과하기 쉽다 —
    # 감사(2026-08) 중 발견해 field_validator(mode='before')로 명시적으로 막았다.
    with pytest.raises(ValidationError):
        CustomTargets(potassium=True)
    with pytest.raises(ValidationError):
        CustomTargets(energy=[True, 2000])


def test_custom_targets_field_present_on_all_four_request_schemas():
    # 이 4곳(GenReq/DayReq/DayTargetsReq/ProfileReq) 모두 없으면(None) 기존과 동일 동작해야 한다.
    for Req in (GenReq, DayReq, DayTargetsReq, ProfileReq):
        req = Req()
        assert req.custom_targets is None


# ---------------------------------------------------------------------------
# FOOK_adjust_levers.day_targets() / meal_bounds() — override 반영 + 하위호환
# ---------------------------------------------------------------------------
def test_day_targets_no_override_matches_original_formula():
    w = F.standard_weight(170, '남')
    d = F.day_targets(w)
    assert d['Elo'] == w * 30 and d['Ehi'] == w * 35
    assert d['Plo'] == pytest.approx(w * 1.1) and d['Phi'] == pytest.approx(w * 1.2)
    assert d['Kmax'] == 3000 and d['Pmax'] == 1000
    assert d['Namax'] == F.SALT_MG * 3
    assert d['Na_total_max'] == F.NA_TOTAL_MEAL * 3   # 기존 meal_bounds()가 쓰던 기본값과 동일


def test_day_targets_override_only_changes_given_keys():
    w = F.standard_weight(170, '남')
    base = F.day_targets(w)
    d = F.day_targets(w, {'potassium': 2200, 'sodium': 1800})
    assert d['Kmax'] == 2200
    assert d['Na_total_max'] == 1800
    # 나머지는 안 건드림
    assert d['Elo'] == base['Elo'] and d['Ehi'] == base['Ehi']
    assert d['Plo'] == base['Plo'] and d['Phi'] == base['Phi']
    assert d['Pmax'] == base['Pmax']
    assert d['Namax'] == base['Namax']   # 첨가염 기준은 sodium override와 무관하게 고정


def test_day_targets_energy_protein_override_sets_lo_and_hi():
    d = F.day_targets(60, {'energy': [1800, 2200], 'protein': [60, 80]})
    assert d['Elo'] == 1800 and d['Ehi'] == 2200
    assert d['Plo'] == 60 and d['Phi'] == 80


def test_meal_bounds_no_override_matches_original_behavior():
    w = F.standard_weight(170, '남')
    b = F.meal_bounds(w, consumed=None, meals_left=3)
    assert b['Namax'] == F.SALT_MG
    assert b['Na_total_target'] == pytest.approx(F.NA_TOTAL_MEAL)  # 남은예산=하루값÷3, 첫 끼


def test_meal_bounds_sodium_override_changes_only_na_total_target():
    w = F.standard_weight(170, '남')
    base = F.meal_bounds(w, consumed=None, meals_left=3)
    overridden = F.meal_bounds(w, consumed=None, meals_left=3, overrides={'sodium': 1800})
    # 첨가염(Namax)은 sodium override와 완전히 무관 — 항상 SALT_MG 고정.
    assert overridden['Namax'] == base['Namax'] == F.SALT_MG
    # 총나트륨 예산만 1800/3=600으로 바뀐다.
    assert overridden['Na_total_target'] == pytest.approx(1800 / 3)
    # 칼륨·인·열량·단백질 등 다른 기준은 그대로.
    for k in ('Elo', 'Ehi', 'Plo', 'Phi', 'Kmax', 'Pmax'):
        assert overridden[k] == base[k]


def test_meal_bounds_potassium_override_propagates_with_remaining_budget_logic():
    w = F.standard_weight(170, '남')
    consumed = {'E': 0, 'protein': 0, 'K': 500, 'P': 0, 'Na': 0}
    b = F.meal_bounds(w, consumed=consumed, meals_left=2, overrides={'potassium': 2400})
    # cap(dv, used) = min((dv-used)/n, fair*(1+tol)) — override된 day 값(2400)을 기준으로 계산돼야 함.
    fair = 2400 / 3
    expected = min((2400 - 500) / 2, fair * 1.2)
    assert b['Kmax'] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# server_FOOK.py / app_core_FOOK.py 배선 확인 (소스 레벨, TF import 없이)
# ---------------------------------------------------------------------------
SERVER_SRC = (BACKEND_DIR / 'server_FOOK.py').read_text(encoding='utf-8')
CORE_SRC = (BACKEND_DIR / 'app_core_FOOK.py').read_text(encoding='utf-8')


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


def test_generate_route_passes_custom_targets():
    src = _extract_function_source(SERVER_SRC, 'generate')
    assert 'custom_targets=_ct(req)' in src


def test_generate_day_route_passes_custom_targets():
    src = _extract_function_source(SERVER_SRC, 'generate_day')
    assert 'custom_targets=_ct(req)' in src


def test_update_profile_route_saves_custom_targets():
    src = _extract_function_source(SERVER_SRC, 'update_profile')
    assert 'custom_targets' in src
    assert 'Jsonb(req.custom_targets' in src
    assert '_ensure_custom_targets_column' in src


def test_me_route_selects_custom_targets():
    src = _extract_function_source(SERVER_SRC, 'me')
    assert 'custom_targets' in src


def test_meal_result_accepts_and_forwards_custom_targets():
    src = _extract_function_source(CORE_SRC, 'meal_result')
    assert 'custom_targets=None' in src
    assert 'overrides=custom_targets' in src


def test_day_result_forwards_custom_targets_to_each_meal():
    src = _extract_function_source(CORE_SRC, 'day_result')
    assert 'custom_targets=custom_targets' in src
    assert "F.day_targets(weight, custom_targets)" in src


def test_meal_result_targets_include_sodium_total_target_without_removing_sodium():
    src = _extract_function_source(CORE_SRC, 'meal_result')
    assert "'sodium': round(b['Namax'])" in src          # 기존 필드 유지(하위호환)
    assert "'sodium_total_target': round(b['Na_total_target'])" in src  # 신규 필드


def test_day_ok_base_condition_still_na_season_only():
    # day_ok의 기본 대입식 자체는 이번 작업(2026-08 2차 수정)에서도 바뀌지 않는다 —
    # 총나트륨은 이 괄호 안이 아니라 그 다음 조건부 줄에서 custom sodium일 때만 추가된다
    # (test_sodium_hard_constraint.py가 그 조건부 동작을 자세히 검증한다).
    src = _extract_function_source(CORE_SRC, 'day_result')
    m = re.search(r"day_ok = \((.*?)\)\n", src, re.DOTALL)
    assert m, 'day_ok 계산식을 찾지 못했습니다.'
    assert "consumed['Na_season'] <= dt['Namax']" in m.group(1)
    assert 'Na_total_max' not in m.group(1)
    # 그 뒤에 custom sodium일 때만 총나트륨을 추가로 보는 조건부 줄이 있어야 한다.
    assert "if dt.get('custom_sodium_active'):" in src
    assert "consumed['Na'] <= dt['Na_total_max']" in src
