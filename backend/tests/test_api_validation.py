"""
가벼운 API 검증 테스트 — app_core_FOOK(TF 모델 로딩, 수십 초 소요)을 import하지 않는다.

server_FOOK.py는 요청 스키마(schemas.py)와 rate limiter(rate_limit.py)를 쓰지만,
server_FOOK.py 자체를 import하면 TF 모델이 통째로 로딩된다. 그래서 여기서는
schemas.py / rate_limit.py를 직접 테스트한다(둘 다 app_core_FOOK을 import하지 않음).
"""
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rate_limit import rate_limit
from schemas import (
    DayReq,
    FindIdReq,
    GenReq,
    LoginReq,
    ProfileReq,
    ResetPasswordReq,
    SignupReq,
)


# ---------------- SignupReq / LoginReq ----------------

def test_signup_valid_username_passes():
    req = SignupReq(username='hong_gildong1', password='password123', name='홍길동')
    assert req.normalized() == 'hong_gildong1'


@pytest.mark.parametrize('bad_username', [
    'ab',                 # 4자 미만 — Field(min_length)에서 바로 막힘
    'a' * 31,             # 30자 초과 — Field(max_length)에서 바로 막힘
    '홍길동',              # 3자라 길이 제약에도 걸리지만, 한글이라 정규식도 어차피 막음
    'user name',          # 길이는 통과하지만 공백이라 normalized()의 정규식에서 막힘
    'user@name',          # 길이는 통과하지만 @ 이 정규식 허용 문자가 아님
])
def test_signup_invalid_username_rejected(bad_username):
    # 길이 위반은 SignupReq(...) 생성 시점(Field 제약)에, 형식 위반(공백/한글/특수문자 등
    # 길이는 유효한 경우)은 그 뒤 normalized()의 정규식 검사에서 걸린다 — 둘 다 결국 막힌다.
    with pytest.raises((ValidationError, ValueError)):
        SignupReq(username=bad_username, password='password123', name='홍길동').normalized()


def test_signup_short_password_rejected():
    with pytest.raises(ValidationError):
        SignupReq(username='validuser1', password='short', name='홍길동')


def test_login_accepts_any_nonempty_strings():
    # 로그인은 형식 검증 없이 통과시키고(존재 여부/비밀번호 일치는 DB 조회에서 판정),
    # 스키마 단에서는 필드가 있는지만 본다.
    req = LoginReq(username='whatever', password='whatever')
    assert req.username == 'whatever'


# ---------------- HeightSexMixin (GenReq/DayReq 공용) ----------------
# 표준체중 계산이 성별에 따라 갈리는데, height만 주고 sex를 빠뜨리면 남성으로 조용히
# 가정되던 버그가 있었다(이번 세션 이전에 고쳐짐) — 이 검증이 그 회귀를 막는다.

def test_height_without_sex_rejected_for_generate():
    with pytest.raises(ValidationError):
        GenReq(height=170)


def test_height_without_sex_rejected_for_day_plan():
    with pytest.raises(ValidationError):
        DayReq(height=170)


@pytest.mark.parametrize('sex', ['남', '여', 'male', 'female'])
def test_height_with_valid_sex_accepted(sex):
    req = GenReq(height=170, sex=sex)
    assert req.height == 170
    assert req.sex == sex


def test_height_with_invalid_sex_value_rejected():
    with pytest.raises(ValidationError):
        GenReq(height=170, sex='남자')  # Literal에 없는 값


def test_no_height_no_sex_is_fine():
    # weight만으로 생성하는 기본 경로(체중 기준 계산) — height/sex 둘 다 없어도 통과해야 한다.
    req = GenReq(weight=60)
    assert req.height is None
    assert req.sex is None


# ---------------- ProfileReq.computed_age ----------------

def test_computed_age_from_valid_birthdate():
    from datetime import date
    req = ProfileReq(birthdate='2000-01-01')
    age = req.computed_age()
    expected = date.today().year - 2000 - ((date.today().month, date.today().day) < (1, 1))
    assert age == expected


def test_computed_age_falls_back_to_explicit_age_field():
    req = ProfileReq(age=45)
    assert req.computed_age() == 45


def test_computed_age_invalid_birthdate_raises():
    req = ProfileReq(birthdate='not-a-date')
    with pytest.raises(ValueError):
        req.computed_age()


def test_profile_height_weight_out_of_range_rejected():
    with pytest.raises(ValidationError):
        ProfileReq(height=10)  # 50 미만
    with pytest.raises(ValidationError):
        ProfileReq(weight=1000)  # 300 초과


# ---------------- GenReq/DayReq.weight (ProfileReq와 동일 범위 재사용) ----------------
# 감사(audit)에서 지적된 항목: GenReq/DayReq는 weight에 하한/상한이 없어서 0·음수·비현실적으로
# 큰 값도 그대로 통과해 표준체중·영양목표 계산에 흘러들어갈 수 있었다. ProfileReq가 이미 쓰는
# 범위(20~300kg)를 그대로 재사용해서 막는다 — 새 기준을 만들지 않는다.

@pytest.mark.parametrize('bad_weight', [0, -5, 1000])
def test_gen_req_weight_out_of_range_rejected(bad_weight):
    with pytest.raises(ValidationError):
        GenReq(weight=bad_weight)


@pytest.mark.parametrize('bad_weight', [0, -5, 1000])
def test_day_req_weight_out_of_range_rejected(bad_weight):
    with pytest.raises(ValidationError):
        DayReq(weight=bad_weight)


def test_gen_req_valid_weight_accepted():
    req = GenReq(weight=60)
    assert req.weight == 60


def test_day_req_valid_weight_accepted():
    req = DayReq(weight=60)
    assert req.weight == 60


def test_day_req_default_weight_is_valid():
    # 기본값(60)이 우리가 새로 건 ge=20,le=300 범위를 벗어나면 안 된다(회귀 방지).
    assert DayReq().weight == 60
    assert GenReq().weight == 60


# ---------------- GenReq.used_today (오늘 이미 기록한 메뉴 → 하루 중복 방지) ----------------
# day_result()는 끼니마다 used_today를 누적해서 make_meal()에 넘겨왔지만(하루 생성),
# /generate 한 끼 경로에는 프론트가 "오늘 실제로 기록한 식사"의 raw_menus를 넘길 방법이
# 없었다. GenReq에 used_today를 추가해 이 경로를 연결한다 — 필드가 없어도(하위호환) 기존
# 요청은 그대로 통과해야 한다.

def test_gen_req_accepts_used_today():
    req = GenReq(used_today=['된장찌개', '현미밥'])
    assert req.used_today == ['된장찌개', '현미밥']


def test_gen_req_used_today_defaults_to_none():
    # 필드를 아예 안 보내는 기존 클라이언트/테스트 요청도 계속 통과해야 한다(하위호환).
    req = GenReq()
    assert req.used_today is None


def test_gen_req_used_today_none_explicit_still_fine():
    req = GenReq(used_today=None)
    assert req.used_today is None


def test_gen_req_used_today_does_not_affect_consumed_validation():
    # used_today와 consumed는 서로 독립된 필드 — 하나를 추가했다고 다른 하나의 기존
    # 검증(체중 범위 등)이 깨지면 안 된다.
    req = GenReq(weight=55, consumed={'E': 100}, used_today=['김치찌개'])
    assert req.weight == 55
    assert req.consumed == {'E': 100}
    assert req.used_today == ['김치찌개']


# ---------------- DayReq.menus/ingredients 길이 (하루 세 끼 고정) ----------------
# 감사에서 지적된 항목: day_result()가 menus[0..2]를 바로 인덱싱해서, 길이 1·2인 menus를
# 보내면 IndexError -> 500이 났다(제대로 된 4xx가 아니었음). 길이 3만 허용하고, None과 빈
# 리스트([])는 '지정 안 함'(기존 server_FOOK.clean()의 의미)으로 그대로 허용해야 한다.

@pytest.mark.parametrize('bad_len_menus', [['된장찌개'], ['된장찌개', '김치찌개'],
                                            ['a', 'b', 'c', 'd']])
def test_day_req_menus_wrong_length_rejected(bad_len_menus):
    with pytest.raises(ValidationError):
        DayReq(menus=bad_len_menus)


@pytest.mark.parametrize('bad_len_ings', [['고등어'], ['고등어', '두부']])
def test_day_req_ingredients_wrong_length_rejected(bad_len_ings):
    with pytest.raises(ValidationError):
        DayReq(ingredients=bad_len_ings)


def test_day_req_menus_length_three_accepted():
    req = DayReq(menus=['된장찌개', None, '고등어조림'])
    assert req.menus == ['된장찌개', None, '고등어조림']


def test_day_req_menus_none_or_empty_still_accepted():
    # None(필드 자체를 안 보냄)과 빈 리스트 둘 다 '지정 안 함'으로 계속 허용되어야 한다
    # (server_FOOK.clean()이 falsy를 None으로 취급해온 기존 동작과의 호환).
    assert DayReq(menus=None).menus is None
    assert DayReq(menus=[]).menus == []


# ---------------- FindIdReq / ResetPasswordReq ----------------

def test_reset_password_requires_min_length_new_password():
    with pytest.raises(ValidationError):
        ResetPasswordReq(username='u', name='홍길동', birthdate='2000-01-01', new_password='short')


def test_find_id_requires_name():
    with pytest.raises(ValidationError):
        FindIdReq(name='', birthdate='2000-01-01')


# ---------------- rate_limit.py ----------------
# gen_batch()가 공유하는 TF 모델을 동시 요청이 건드리는 문제(레이스)와는 별개로,
# /login·/recipe·/tts처럼 비용이 있거나 무차별 대입에 노출된 엔드포인트를 위한 제한이다.
# 실제 server_FOOK.py를 띄우지 않고, 같은 rate_limit() 데코레이터를 쓰는 작은 앱으로 검증한다.

def _make_limited_app(max_requests: int, window_seconds: int, prefix: str) -> FastAPI:
    # rate_limit._HITS는 모듈 전역이라 프로세스(=pytest 세션) 내내 살아있다.
    # 테스트마다 key_prefix를 다르게 줘서 서로 카운트가 섞이지 않게 한다.
    app = FastAPI()

    @app.get('/ping', dependencies=[Depends(rate_limit(prefix, max_requests, window_seconds))])
    def ping():
        return {'ok': True}

    return app


def test_rate_limit_allows_up_to_max_requests():
    client = TestClient(_make_limited_app(3, 60, 'allow-up-to-max'))
    for _ in range(3):
        r = client.get('/ping')
        assert r.status_code == 200


def test_rate_limit_blocks_after_max_requests():
    client = TestClient(_make_limited_app(3, 60, 'blocks-after-max'))
    for _ in range(3):
        assert client.get('/ping').status_code == 200
    r = client.get('/ping')
    assert r.status_code == 429
    assert 'Retry-After' in r.headers


def test_rate_limit_is_per_client_ip():
    # TestClient는 기본적으로 같은 IP(testclient)로 요청하므로, X-Forwarded-For로
    # 다른 클라이언트를 흉내내면 그 클라이언트는 별도 한도를 받아야 한다.
    client = TestClient(_make_limited_app(1, 60, 'per-client-ip'))
    assert client.get('/ping', headers={'x-forwarded-for': '1.1.1.1'}).status_code == 200
    assert client.get('/ping', headers={'x-forwarded-for': '1.1.1.1'}).status_code == 429
    assert client.get('/ping', headers={'x-forwarded-for': '2.2.2.2'}).status_code == 200


def test_rate_limit_window_resets_after_time_passes(monkeypatch):
    import rate_limit as rl

    t = [1_000_000.0]
    monkeypatch.setattr(rl.time, 'time', lambda: t[0])

    client = TestClient(_make_limited_app(1, 10, 'window-resets'))
    assert client.get('/ping', headers={'x-forwarded-for': '9.9.9.9'}).status_code == 200
    assert client.get('/ping', headers={'x-forwarded-for': '9.9.9.9'}).status_code == 429
    t[0] += 11  # 윈도우(10초)를 넘겨서 진행
    assert client.get('/ping', headers={'x-forwarded-for': '9.9.9.9'}).status_code == 200
