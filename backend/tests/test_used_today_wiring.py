"""
GenReq.used_today가 실제로 /generate 라우트 핸들러에서 core.meal_result(used_today=...)로
전달되는지 확인한다.

server_FOOK.py를 통째로 import하면 모듈 최상단의 `import app_core_FOOK as core`가 TF 모델을
로딩한다(test_consumed_validation_shared.py 상단 설명과 동일한 이유로 다른 테스트들이 피하는
비용). 그래서 여기서도 같은 패턴을 따른다 — generate() 함수의 실제 소스를 그대로 잘라내
문자열 수준에서 배선을 확인한다(재구현이 아니라 원본 텍스트 검사이므로, 실제로 인자 이름이
바뀌거나 호출이 빠지면 이 테스트가 깨진다).
"""
import re
from pathlib import Path

SERVER_SRC_PATH = Path(__file__).resolve().parent.parent / 'server_FOOK.py'
SERVER_SRC = SERVER_SRC_PATH.read_text(encoding='utf-8')


def _extract_function_source(src: str, name: str) -> str:
    m = re.search(rf'^def {re.escape(name)}\(', src, re.MULTILINE)
    assert m, f'{name}() 정의를 server_FOOK.py에서 찾지 못했습니다.'
    lines = src[m.start():].splitlines(keepends=True)
    out = [lines[0]]
    for line in lines[1:]:
        if line.strip() == '' or line[:1] in (' ', '\t'):
            out.append(line)
        else:
            break
    return ''.join(out)


GENERATE_SRC = _extract_function_source(SERVER_SRC, 'generate')


def test_generate_route_reads_used_today_from_request():
    assert 'req.used_today' in GENERATE_SRC, (
        'generate()가 req.used_today를 읽지 않습니다 — GenReq에 필드만 추가되고 '
        '실제로는 쓰이지 않는 회귀를 막기 위한 확인.'
    )


def test_generate_route_passes_used_today_to_meal_result():
    assert re.search(r'used_today\s*=\s*used_today', GENERATE_SRC), (
        'generate()가 core.meal_result(...)에 used_today를 넘기지 않습니다.'
    )


def test_generate_route_still_calls_meal_result_with_existing_args():
    # used_today 배선을 추가하면서 기존 인자(consumed/meals_left 등) 전달이 실수로
    # 지워지지 않았는지 확인 — day_result()의 기존 동작(터치 금지 대상)과는 무관하게,
    # /generate 자체의 회귀만 본다.
    for arg in ('menu=', 'ingredient=', 'weight=req.weight', 'consumed=consumed',
                'meals_left='):
        assert arg in GENERATE_SRC, f'generate()에서 {arg} 전달이 사라졌습니다.'


def test_generate_day_route_untouched_day_result_call():
    # day_result()는 이미 자체적으로 used_today를 하루 동안 누적한다(app_core_FOOK.py).
    # /generate_day 핸들러 쪽 배선은 이번 작업 대상이 아니므로 그대로인지만 확인한다.
    day_src = _extract_function_source(SERVER_SRC, 'generate_day')
    assert 'core.day_result(' in day_src
    assert 'used_today' not in day_src  # day_result가 내부에서 알아서 처리 — 라우트는 몰라도 됨
