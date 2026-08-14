"""
server_FOOK.parse_consumed()의 음수/NaN/inf 거부가 /generate·/chat 양쪽에 실제로
적용되는지 확인한다.

server_FOOK.py를 통째로 import하면 모듈 최상단의 `import app_core_FOOK as core`가 TF
모델을 로딩한다 — 다른 테스트 파일들이 전부 피하는 비용이자 CI가 일부러 TensorFlow를
requirements-dev.txt에서 빼놓은 이유다(test_auth_endpoints.py 상단 설명 참고). 그래서
두 갈래로 검증한다:

1. parse_consumed() 함수 본문을 소스에서 그대로 잘라내(재구현이 아니라 원본 텍스트) 가벼운
   네임스페이스(math, fastapi.HTTPException만 필요)에서 실행해서 실제 검증 로직 자체를
   진짜로 확인한다.
2. /generate·/chat 라우트 핸들러가 각자 자기 본문에서 `parse_consumed(req.consumed)`를
   호출한다는 것, 그리고 parse_consumed 정의가 파일 안에 단 하나뿐이라는 것을 소스 검사로
   확인해서 "두 라우트가 진짜 같은 함수 하나를 공유한다"(복사본이 아니다)를 구조적으로
   보장한다.
"""
import math
import re
from pathlib import Path

import pytest
from fastapi import HTTPException

SERVER_SRC_PATH = Path(__file__).resolve().parent.parent / 'server_FOOK.py'
SERVER_SRC = SERVER_SRC_PATH.read_text(encoding='utf-8')

CONSUMED_KEYS = ('E', 'protein', 'K', 'P', 'Na', 'Na_season')


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


def _load_parse_consumed():
    consumed_keys_line = re.search(r'^CONSUMED_KEYS = .*$', SERVER_SRC, re.MULTILINE)
    nutrition_keys_line = re.search(r'^NUTRITION_KEYS = .*$', SERVER_SRC, re.MULTILINE)
    assert consumed_keys_line and nutrition_keys_line, 'CONSUMED_KEYS/NUTRITION_KEYS 정의를 찾지 못했습니다.'
    ns = {'math': math, 'HTTPException': HTTPException}
    exec(consumed_keys_line.group(0), ns)
    exec(nutrition_keys_line.group(0), ns)
    exec(_extract_function_source(SERVER_SRC, 'parse_consumed'), ns)
    return ns['parse_consumed']


parse_consumed = _load_parse_consumed()


def _consumed_with(key, value):
    return {k: (value if k == key else 0) for k in CONSUMED_KEYS}


# ---------------- 1. 실제 함수 로직: 음수/NaN/inf 거부, 0 허용, 숫자 아님 거부 ----------------

BAD_VALUES = [-1, -0.5, math.nan, math.inf, -math.inf]


@pytest.mark.parametrize('bad_value', BAD_VALUES, ids=['neg_int', 'neg_float', 'nan', 'inf', 'neg_inf'])
def test_parse_consumed_rejects_negative_nan_inf(bad_value):
    with pytest.raises(HTTPException) as exc:
        parse_consumed(_consumed_with('K', bad_value))
    assert exc.value.status_code == 422


def test_parse_consumed_allows_zero():
    # 0은 "아직 안 먹음"이라는 정당한 값이라 계속 허용돼야 한다(회귀 방지).
    result = parse_consumed(_consumed_with('K', 0))
    assert result['K'] == 0


def test_parse_consumed_rejects_non_numeric():
    with pytest.raises(HTTPException) as exc:
        parse_consumed(_consumed_with('K', 'not-a-number'))
    assert exc.value.status_code == 422


def test_parse_consumed_none_passthrough():
    # raw가 falsy(None 등)면 검증 없이 None을 그대로 돌려준다 — consumed 없이 요청하는
    # 기존 정상 경로(비개인화 질문/생성)를 깨지 않기 위한 동작.
    assert parse_consumed(None) is None
    assert parse_consumed({}) is None


# ---------------- 2. 구조: /generate·/chat이 이 함수 하나를 실제로 공유하는지 ----------------

def test_parse_consumed_defined_exactly_once():
    # 두 라우트가 각자 다른 사본을 갖고 있어서 한쪽만 고쳐지는 회귀를 막기 위한 확인.
    assert len(re.findall(r'^def parse_consumed\(', SERVER_SRC, re.MULTILINE)) == 1


@pytest.mark.parametrize('route_name', ['generate', 'chat'])
def test_route_calls_shared_parse_consumed(route_name):
    body = _extract_function_source(SERVER_SRC, route_name)
    assert 'parse_consumed(req.consumed)' in body, (
        f'{route_name}()가 parse_consumed(req.consumed)를 호출하지 않습니다 — '
        f'/generate와 /chat이 검증 로직을 공유한다는 전제가 깨졌습니다.'
    )
