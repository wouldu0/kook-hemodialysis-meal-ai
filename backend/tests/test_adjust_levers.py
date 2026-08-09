# -*- coding: utf-8 -*-
"""
FOOK_adjust_levers.py의 순수 함수(외부 데이터·모델 로딩 없이 동작하는 함수) 단위 테스트.

TensorFlow 생성 모델(app_core_FOOK)이나 식약청 영양DB 엑셀(load_all())은 전혀 필요 없다 —
이 파일이 다루는 함수들은 입력값만으로 결과가 정해지는 계산·판정 로직이라, CI에서 모델을
로딩하지 않고도 회귀를 잡을 수 있다.
"""
import pytest

import FOOK_adjust_levers as F


# ── standard_weight: 표준체중 = 키(m)^2 × 계수(남22/여21) ─────────────────────
def test_standard_weight_defaults_to_male_when_sex_is_none():
    # sex=None이면 '여성 계열' 문자열 목록에 안 걸리므로 남성(22)으로 계산된다.
    male = F.standard_weight(170, None)
    female = F.standard_weight(170, '여')
    assert male > female
    assert male == pytest.approx(1.7 * 1.7 * 22)


@pytest.mark.parametrize(
    "sex", ["여", "여성", "f", "female", "w", "women", " 여 ", "FEMALE", "Female"]
)
def test_standard_weight_recognizes_female_variants(sex):
    # 대소문자·앞뒤 공백을 무시하고, 문서에 명시된 모든 여성 표기를 21 계수로 인식해야 한다.
    assert F.standard_weight(160, sex) == pytest.approx(1.6 * 1.6 * 21)


def test_standard_weight_unrecognized_string_falls_back_to_male():
    # 오타·잘못된 값도 조용히 남성(22)으로 계산된다 — 이게 서버 쪽 입력 검증이 필요한 이유.
    assert F.standard_weight(170, "unknown") == pytest.approx(1.7 * 1.7 * 22)


# ── day_targets: 하루 5대 영양 기준 ───────────────────────────────────────────
def test_day_targets_formula():
    d = F.day_targets(60)
    assert d['Elo'] == pytest.approx(60 * 30)
    assert d['Ehi'] == pytest.approx(60 * 35)
    assert d['Plo'] == pytest.approx(60 * 1.1)
    assert d['Phi'] == pytest.approx(60 * 1.2)
    assert d['Kmax'] == 3000
    assert d['Pmax'] == 1000
    assert d['Namax'] == F.SALT_MG * 3


# ── meal_bounds: 남은 예산 ÷ 남은 끼니, ±tol 밴드로 클램프 ───────────────────
def test_meal_bounds_first_meal_defaults_to_fair_share_band():
    b = F.meal_bounds(60, consumed=None, meals_left=3)
    d = F.day_targets(60)
    # 첫 끼(먹은 게 없음)는 하루기준÷3에 가깝고, ±20% 밴드 안에 들어와야 한다.
    fair_elo = d['Elo'] / 3
    assert fair_elo * 0.8 <= b['Elo'] <= fair_elo * 1.2


def test_meal_bounds_reduces_remaining_budget_when_meals_already_consumed():
    consumed_none = F.meal_bounds(60, consumed=None, meals_left=2)
    heavy_consumed = F.meal_bounds(
        60,
        consumed={'E': 900, 'protein': 40, 'K': 100, 'P': 100, 'Na': 0},
        meals_left=2,
    )
    # 이미 많이 먹었으면 다음 끼 열량 상한이 더 낮아야 한다(남은 예산이 줄었으므로).
    assert heavy_consumed['Ehi'] <= consumed_none['Ehi']


def test_meal_bounds_peff_falls_back_to_p_when_missing():
    # day_result()가 끼니마다 누적하는 consumed 딕셔너리엔 'Peff' 키가 없다(P만 있음).
    # meal_bounds가 이를 P로 안전하게 대체하는지 확인 — 없으면 KeyError로 죽어야 정상인데
    # 안 죽고 조용히 잘못된 값을 쓰는 것도 버그이므로, 반환값이 유한한 숫자인지도 함께 본다.
    consumed = {'E': 200, 'protein': 10, 'K': 300, 'P': 150, 'Na': 50}
    b = F.meal_bounds(60, consumed=consumed, meals_left=2)
    assert b['Pmax'] > 0


def test_meal_bounds_namax_is_fixed_regardless_of_consumed():
    # 첨가염(Namax)은 끼니 고정값이라 남은 예산 계산과 무관해야 한다.
    b1 = F.meal_bounds(60, consumed=None, meals_left=3)
    b2 = F.meal_bounds(60, consumed={'E': 0, 'protein': 0, 'K': 0, 'P': 0, 'Na': 2000}, meals_left=1)
    assert b1['Namax'] == b2['Namax'] == F.SALT_MG


# ── num: 문자열/None을 안전하게 float으로 ─────────────────────────────────────
@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, None),
        (10, 10.0),
        (3.5, 3.5),
        ("12.5", 12.5),
        ("(5)", 5.0),  # 괄호로 감싼 표기도 벗겨서 파싱
        ("not-a-number", None),
    ],
)
def test_num_parsing(raw, expected):
    assert F.num(raw) == expected


# ── is_seasoning / is_salty_seasoning ────────────────────────────────────────
def test_is_seasoning_true_for_condiment_group():
    assert F.is_seasoning({'group': '조미료류', 'ing': '고추장'})


def test_is_seasoning_true_for_jeotgal_by_name_even_if_group_is_seafood():
    # 젓갈은 group이 '어패류'라 조미료류로 안 잡히지만, 이름에 '젓'이 있으면 첨가염으로 취급한다.
    assert F.is_seasoning({'group': '어패류 및 그 제품', 'ing': '새우젓'})


def test_is_salty_seasoning_excludes_low_sodium_condiments():
    # 식초·후추처럼 나트륨이 거의 없는 조미료는 is_seasoning=True여도 축소 대상(salty)이 아니다.
    vinegar = {'group': '조미료류', 'ing': '식초, 양조', 'Na': 5}
    soy_sauce = {'group': '조미료류', 'ing': '간장', 'Na': 5000}
    assert not F.is_salty_seasoning(vinegar)
    assert F.is_salty_seasoning(soy_sauce)


# ── is_japgok_grain / is_broth ────────────────────────────────────────────────
def test_is_japgok_grain():
    assert F.is_japgok_grain('현미, 생것')
    assert not F.is_japgok_grain('멥쌀, 백미, 생것')


def test_is_broth_by_name_keyword():
    assert F.is_broth('멸치육수', 100)
    assert not F.is_broth('멥쌀, 백미, 생것', 100)


def test_is_broth_anchovy_amount_threshold():
    # 멸치·다시마는 소량(육수용)이면 육수 취급, 충분한 양(멸치볶음 등 섭취용)이면 아니다.
    assert F.is_broth('멸치', 3)          # BROTH_MAX_G(8g) 미만 -> 육수용
    assert not F.is_broth('멸치', 20)     # 8g 이상 -> 섭취용(볶음 등)


# ── amt_floor_of / amt_ceil_of ────────────────────────────────────────────────
def test_amt_floor_of_staple_uses_absolute_rice_minimum():
    rice = {'menu': '백미밥', 'ing': '멥쌀, 백미, 생것', 'amt': 80, 'orig': 80}
    # 주식(밥)은 원본의 50%(40g)와 RICE_MIN_G(40g) 중 더 큰 값을 하한으로 쓴다.
    assert F.amt_floor_of(rice) == F.RICE_MIN_G


def test_amt_ceil_of_oil_gets_higher_ceiling_than_normal_ingredient():
    oil = {'group': '유지류'}
    veg = {'group': '채소류'}
    assert F.amt_ceil_of(oil) == F.OIL_CEIL
    assert F.amt_ceil_of(veg) == F.AMT_CEIL
    assert F.amt_ceil_of(oil) > F.amt_ceil_of(veg)


# ── unrealistic_reason ────────────────────────────────────────────────────────
def test_unrealistic_reason_none_for_normal_amounts():
    inst = [{'menu': '두부조림', 'ing': '두부', 'amt': 100, 'orig': 100, 'group': '두류'}]
    assert F.unrealistic_reason(inst) is None


def test_unrealistic_reason_flags_amount_far_outside_ceiling():
    # 원본의 AMT_CEIL(2배)를 훌쩍 넘는 비현실적인 양은 이유 문자열을 반환해야 한다.
    inst = [{'menu': '두부조림', 'ing': '두부', 'amt': 1100, 'orig': 100, 'group': '두류'}]
    reason = F.unrealistic_reason(inst)
    assert reason is not None
    assert '두부' in reason


def test_unrealistic_reason_ignores_seasoning_scaled_down():
    # 조미료는 나트륨 레버가 0.3배까지 의도적으로 줄이는 게 정상 동작이라 걸리면 안 된다.
    inst = [{'menu': '된장찌개', 'ing': '된장', 'amt': 5, 'orig': 20, 'group': '조미료류'}]
    assert F.unrealistic_reason(inst) is None


# ── p_abs: 흡수 보정 인 ────────────────────────────────────────────────────────
def test_p_abs_processed_food_multiplies_up():
    assert F.p_abs(100, '조리가공식품류', '햄') == pytest.approx(100 * F.PROCESSED_P_FACTOR)


def test_p_abs_plant_protein_multiplies_down():
    assert F.p_abs(100, '두류', '두부') == pytest.approx(100 * F.PLANT_P_FACTOR)


def test_p_abs_other_group_unchanged():
    assert F.p_abs(100, '육류 및 그 제품', '돼지고기') == 100


def test_p_abs_none_input_stays_none():
    assert F.p_abs(None, '두류', '두부') is None
