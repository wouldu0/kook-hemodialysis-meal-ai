"""
새로 추가한 /day_targets 라우트와 core.day_targets_only()를 확인한다.

server_FOOK.py/app_core_FOOK.py를 통째로 import하면 TF 모델 로딩 비용이 들어서
(다른 test_*_route*.py와 같은 이유), 여기서도 같은 패턴을 따른다 — 함수 소스를
그대로 잘라내 배선을 확인하고, 순수 계산 부분(day_targets_only가 기대는
FOOK_adjust_levers.standard_weight/day_targets)만 별도로 실행해 값을 검증한다.
"""
import ast
import re
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
SERVER_SRC = (BACKEND_DIR / 'server_FOOK.py').read_text(encoding='utf-8')
CORE_SRC = (BACKEND_DIR / 'app_core_FOOK.py').read_text(encoding='utf-8')
LEVERS_SRC = (BACKEND_DIR / 'FOOK_adjust_levers.py').read_text(encoding='utf-8')


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


ROUTE_SRC = _extract_function_source(SERVER_SRC, 'day_targets')
CORE_FN_SRC = _extract_function_source(CORE_SRC, 'day_targets_only')


def test_route_calls_day_targets_only_with_profile_fields():
    assert 'core.day_targets_only(' in ROUTE_SRC
    for arg in ('weight=req.weight', 'height=req.height', 'sex=req.sex', 'custom_targets=_ct(req)'):
        assert arg in ROUTE_SRC, f'/day_targets 라우트가 {arg} 전달을 빠뜨렸습니다.'


def test_route_does_not_call_meal_generation():
    # 메뉴 생성을 실수로 함께 돌리면 이 엔드포인트를 가볍게 만든 의미가 없어진다.
    assert 'meal_result' not in ROUTE_SRC
    assert 'day_result(' not in ROUTE_SRC


def test_day_targets_only_uses_standard_weight_when_height_given():
    assert 'F.standard_weight(height, sex)' in CORE_FN_SRC


def test_day_targets_only_values_match_formula():
    # day_targets_only()가 실제로 계산에 쓰는 두 순수 함수(F.standard_weight, F.day_targets)를
    # 원본 그대로 실행해서, day_targets_only()가 반환하는 필드/반올림 규칙이 일치하는지 검증한다.
    tree = ast.parse(LEVERS_SRC)
    funcs = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ('standard_weight', 'day_targets'):
            funcs[node.name] = ast.get_source_segment(LEVERS_SRC, node)
    ns = {'SALT_MG': 393, 'NA_TOTAL_MEAL': 655}
    exec(funcs['standard_weight'], ns)
    exec(funcs['day_targets'], ns)

    def day_targets_only(weight=60, height=None, sex=None, custom_targets=None):
        if height is not None:
            weight = ns['standard_weight'](height, sex)
        d = ns['day_targets'](weight, custom_targets)
        return {'energy': [round(d['Elo']), round(d['Ehi'])],
                'protein': [round(d['Plo'], 1), round(d['Phi'], 1)],
                'potassium': round(d['Kmax']),
                'phosphorus': round(d['Pmax']),
                'sodium': round(d['Namax']),
                'sodium_total_target': round(d['Na_total_max'])}

    out = day_targets_only(height=170, sex='남')
    assert out == {
        'energy': [1907, 2225],
        'protein': [69.9, 76.3],
        'potassium': 3000,
        'phosphorus': 1000,
        'sodium': 1179,
        'sodium_total_target': 1965,
    }

    # height 없이 weight만 준 경우(비회원 체험 등) — standard_weight를 거치지 않고 weight를 그대로 쓴다.
    out_no_height = day_targets_only(weight=60)
    assert out_no_height['energy'] == [1800, 2100]
    assert out_no_height['sodium'] == 1179
    assert out_no_height['sodium_total_target'] == 1965

    # custom_targets로 칼륨·총나트륨만 override — 나머지는 자동 산출값 그대로, Namax(첨가염)는
    # sodium override와 무관하게 고정.
    out_custom = day_targets_only(height=170, sex='남',
                                   custom_targets={'potassium': 2200, 'sodium': 1800})
    assert out_custom['potassium'] == 2200
    assert out_custom['sodium_total_target'] == 1800
    assert out_custom['sodium'] == 1179          # 첨가염 기준은 그대로
    assert out_custom['energy'] == out['energy']  # 안 건드린 항목은 자동 산출값 유지
