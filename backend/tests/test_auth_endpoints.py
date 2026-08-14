"""
/auth/find-id, /auth/reset-password 검증.

server_FOOK.py 모듈 최상단은 `import app_core_FOOK as core`로 TF 모델을 로딩하고
(database.py는 sqlalchemy, auth_utils.py도 sqlalchemy가 필요하다). 다른 테스트 파일들
(test_api_validation.py, test_rag_chatbot.py)이 server_FOOK.py를 아예 import하지 않는
이유가 바로 이거고, backend/requirements-dev.txt와 .github/workflows/ci.yml이 일부러
TensorFlow/sqlalchemy/psycopg를 CI 의존성에서 빼놓은 이유이기도 하다(주석 참고) — 여기서
그걸 깨고 무거운 의존성을 새로 추가하고 싶지 않다.

그래서 이 파일은 server_FOOK.py를 import하지 않고 두 갈래로 검증한다:

1. find_id()/reset_password() 함수의 실제 소스 코드를 파일에서 그대로 잘라내(재구현이
   아니라 원본 텍스트 그대로) 가벼운 가짜 db/text/_ensure_birthdate_column과 함께 그
   네임스페이스에서 실행한다 — 실제 함수 로직(통합 404, 세션 revoke)을 진짜로 검증하되
   무거운 모듈은 하나도 import하지 않는다.
2. 레이트리밋 데코레이터가 실제로 어떻게 걸려 있는지는 소스 문자열로 직접 확인하고,
   그 설정값(5회/900초)이 실제로 그렇게 동작하는지는 rate_limit.py(가벼운 순수 모듈)를
   같은 값으로 직접 구동해 확인한다.
"""
import re
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from rate_limit import rate_limit
from schemas import FindIdReq, ResetPasswordReq

SERVER_SRC_PATH = Path(__file__).resolve().parent.parent / 'server_FOOK.py'
SERVER_SRC = SERVER_SRC_PATH.read_text(encoding='utf-8')


def _extract_function_source(src: str, name: str) -> str:
    """`def {name}(...):`부터, 들여쓰기 없는 다음 줄(다음 top-level 정의) 직전까지 잘라낸다."""
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


class _FakeResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def mappings(self):
        return self

    def all(self):
        return self._rows

    def first(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    """select 계열 쿼리는 미리 정해둔 select_rows를, 그 외(UPDATE 등)는 빈 결과를 준다.
    실행된 (sql, params)를 executed에 쌓아서 "정말로 그 UPDATE가 나갔는지"까지 확인한다.
    """

    def __init__(self, select_rows=()):
        self.select_rows = list(select_rows)
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if sql.strip().lower().startswith('select'):
            return _FakeResult(self.select_rows)
        return _FakeResult([])


class _FakeDB:
    def __init__(self, select_rows=()):
        self.conn = _FakeConn(select_rows)

    def __call__(self):
        return self

    def __enter__(self):
        return self.conn

    def __exit__(self, *exc):
        return False


def _build_namespace():
    """find_id()/reset_password()가 참조하는 이름들을, 무거운 모듈 import 없이 채운
    실행 네임스페이스. `db`는 함수 호출 직전에 테스트마다 따로 주입한다.
    """
    ns = {
        'HTTPException': HTTPException,
        'text': lambda s: s,  # sqlalchemy.text는 여기선 그냥 SQL 문자열을 그대로 통과시키는 것으로 대체
        'hash_password': lambda pw: f'hashed:{pw}',  # auth_utils.hash_password 대체(해싱 자체는 이 테스트의 관심사가 아님)
        # 원본 파일은 `from __future__ import annotations`라 타입힌트가 지연 평가되지만,
        # 여기서는 함수 본문만 따로 exec하므로 힌트에 쓰인 타입이 즉시 평가된다 — 실제로
        # 존재하는 요청 스키마를 넣어준다.
        'FindIdReq': FindIdReq,
        'ResetPasswordReq': ResetPasswordReq,
    }
    # _parse_birthdate()는 원본 소스 그대로 실행(날짜 파싱 로직을 재구현하지 않기 위해)
    exec(_extract_function_source(SERVER_SRC, '_parse_birthdate'), ns)
    # _ensure_birthdate_column()은 DDL 캐시 로직이라 이 테스트와 무관 — no-op으로 대체
    ns['_ensure_birthdate_column'] = lambda conn: None
    exec(_extract_function_source(SERVER_SRC, 'find_id'), ns)
    exec(_extract_function_source(SERVER_SRC, 'reset_password'), ns)
    return ns


@pytest.fixture
def ns():
    return _build_namespace()


# ---------------- 통합 404: "계정 없음"과 "정보 불일치"를 구분하지 않는다(계정 열거 방지) ----------------

def test_find_id_unified_404_when_no_match(ns):
    # 쿼리 자체가 이름 AND 생년월일 조건이라, "그런 이름 자체가 없음"과 "이름은 맞는데
    # 생년월일이 틀림"은 코드 입장에서 구분할 방법이 없다 — 둘 다 그냥 rows=0이다.
    ns['db'] = _FakeDB(select_rows=())
    with pytest.raises(HTTPException) as exc:
        ns['find_id'](FindIdReq(name='없는사람', birthdate='2000-01-01'))
    assert exc.value.status_code == 404
    assert exc.value.detail == '입력하신 이름과 생년월일로 가입된 아이디를 찾지 못했습니다.'


def test_find_id_success_when_match(ns):
    from datetime import datetime
    ns['db'] = _FakeDB(select_rows=[{'email': 'a@b.com', 'created_at': datetime(2024, 1, 1)}])
    result = ns['find_id'](FindIdReq(name='홍길동', birthdate='2000-01-01'))
    assert result['usernames'] == ['a@b.com']


def test_reset_password_unified_404_when_no_match(ns):
    ns['db'] = _FakeDB(select_rows=())
    with pytest.raises(HTTPException) as exc:
        ns['reset_password'](ResetPasswordReq(
            username='nouser@example.com', name='없는사람', birthdate='2000-01-01',
            new_password='newpassword123'))
    assert exc.value.status_code == 404
    assert exc.value.detail == '아이디·이름·생년월일이 모두 일치하는 계정을 찾지 못했습니다.'


def test_reset_password_success_revokes_existing_sessions(ns):
    # 코드 주석("비밀번호가 바뀌었으니 기존에 로그인돼 있던 세션은 모두 끊는다")이 실제로
    # 지켜지는지 확인 — 성공 응답뿐 아니라 실행된 SQL까지 검사한다.
    fake = _FakeDB(select_rows=[{'id': 42}])
    ns['db'] = fake
    result = ns['reset_password'](ResetPasswordReq(
        username='user@example.com', name='홍길동', birthdate='2000-01-01',
        new_password='newpassword123'))
    assert result == {'ok': True}

    executed = fake.conn.executed
    password_updates = [(s, p) for s, p in executed if 'update app_users set password_hash' in s.lower()]
    revoke_updates = [(s, p) for s, p in executed if 'update auth_sessions set revoked_at' in s.lower()]
    assert len(password_updates) == 1
    assert len(revoke_updates) == 1
    # revoke 대상이 방금 비밀번호를 바꾼 그 유저(id=42)여야 한다.
    assert revoke_updates[0][1] == {'i': 42}


def test_reset_password_source_comment_still_documents_session_revoke():
    # 코드 자체가 바뀌지 않았어도, 이번 세션의 편집 과정에서 그 설명 주석이 실수로
    # 지워지지 않았는지 확인한다(주석은 동작에 영향 없지만, 이 안전장치의 존재 이유를
    # 남겨두는 문서라 회귀를 잡아둘 가치가 있다).
    body = _extract_function_source(SERVER_SRC, 'reset_password')
    assert '세션' in body and 'revoke' in body.lower()


# ---------------- 레이트리밋: find-id가 reset-password와 동일한 설정(5회/900초)으로 걸려 있는지 ----------------

def test_find_id_and_reset_password_have_matching_rate_limit_wiring_in_source():
    # 실제 데코레이터 문자열을 직접 확인 — 숫자가 바뀌거나 데코레이터 자체가 빠지면
    # 이 테스트가 바로 잡아낸다.
    assert ("@app.post('/auth/find-id', dependencies=[Depends(rate_limit('find-id', 5, 900))])"
            in SERVER_SRC)
    assert ("@app.post('/auth/reset-password', "
            "dependencies=[Depends(rate_limit('reset-password', 5, 900))])" in SERVER_SRC)


def _make_limited_app(max_requests: int, window_seconds: int, prefix: str) -> FastAPI:
    app = FastAPI()

    @app.get('/ping', dependencies=[Depends(rate_limit(prefix, max_requests, window_seconds))])
    def ping():
        return {'ok': True}

    return app


@pytest.mark.parametrize('prefix', ['find-id', 'reset-password'])
def test_rate_limit_behaves_correctly_at_production_config(prefix):
    # server_FOOK.py가 실제로 쓰는 값(5회/900초)을 그대로 rate_limit()에 넣어 구동해서,
    # 위 소스 검사에서 확인한 그 설정이 실제로 "5번까지 통과, 6번째부터 429"로 동작하는지
    # 확인한다. test_api_validation.py의 test_rate_limit_blocks_after_max_requests와 같은
    # 패턴이며, key_prefix만 실제 운영값(find-id/reset-password)으로 맞췄다.
    client = TestClient(_make_limited_app(5, 900, f'prod-config-{prefix}'))
    headers = {'x-forwarded-for': f'{prefix}-prod-config-test'}
    for _ in range(5):
        assert client.get('/ping', headers=headers).status_code == 200
    r = client.get('/ping', headers=headers)
    assert r.status_code == 429
    assert 'Retry-After' in r.headers
