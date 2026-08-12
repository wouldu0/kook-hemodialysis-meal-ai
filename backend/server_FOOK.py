# -*- coding: utf-8 -*-
"""
server_FOOK.py — KOOK 백엔드 API (FastAPI)

한 서버에서 두 축을 함께 처리합니다.
  - 식단 생성(/generate, /generate_day), 조리법 편집(/recipe), 음성 변환(/tts),
    메뉴/재료 목록(/menus, /ingredients) → KLUE-BERT + REINFORCE 강화학습 엔진
    (app_core_FOOK.py / recipe_editor_FOOK.py)
  - 회원가입/로그인/세션/프로필/식단기록/즐겨찾기/PDF보관함/장바구니 → Neon(PostgreSQL) DB
    (database.py / auth_utils.py / SQL 스키마)

실행:
  conda activate foodbert   (또는 requirements.txt로 venv 구성)
  set TF_USE_LEGACY_KERAS=1        (mac/linux는 export)
  python -m uvicorn server_FOOK:app --host 0.0.0.0 --port 8000

주의:
  - 첫 실행 시 app_core_FOOK import 과정에서 TF 모델 로딩에 수십 초가 걸립니다.
  - DATABASE_URL(.env)이 없으면 회원 관련 API(/auth/*, /me/*)는 500을 반환하지만,
    /generate 등 AI 생성 기능은 DB 없이도 정상 동작합니다(서로 독립적인 의존성).
  - /recipe, /tts는 OPENAI_API_KEY가 있어야 동작합니다. 없으면 각각 error 필드 응답 / 400.
"""
from __future__ import annotations
import os, uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy import text
from psycopg.types.json import Jsonb

from database import db
from auth_utils import hash_password, verify_password, issue_session, resolve_user
from rate_limit import rate_limit
from schemas import (
    CartReq, ChatReq, DayReq, FindIdReq, GenReq, LoginReq,
    ProfileReq, RecipeReq, ResetPasswordReq, SaveReq, SignupReq, TTSReq,
)

# ── AI 생성 엔진 로딩 (import 시 TF 모델 1회 로딩 — 수십 초 소요) ──────────────
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
# app_core_FOOK / FOOK_adjust_levers / recipe_editor_FOOK 모두 자기 파일 위치 기준
# 상대경로로 data/, Diet-Generation-.../, Exploiting-.../ 를 읽으므로, 어디서 uvicorn을
# 실행 위치와 무관하게 backend/ 를 작업 디렉터리로 고정한다.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import app_core_FOOK as core

app = FastAPI(title='FOOK 통합 API', version='10.0.0')
# 실제 배포 프론트 주소만 명시적으로 허용한다. 예전엔 *.vercel.app 전체를 정규식으로
# 열어뒀는데(미리보기 배포 주소가 매번 바뀌는 걸 신경 안 쓰려는 목적), 그러면 이 API가
# 임의의 다른 Vercel 프로젝트에서도 크레덴셜 포함 요청을 보낼 수 있게 돼 필요 이상으로 넓다.
# 미리보기 배포를 테스트해야 하면 Render 환경변수 CORS_ORIGIN_REGEX로 그때만 열면 된다.
origins = [x.strip() for x in os.getenv(
    'CORS_ORIGINS',
    'https://kook-hemodialysis-meal-ai.vercel.app,http://localhost:5173,http://127.0.0.1:5173'
).split(',') if x.strip()]
origin_regex = os.getenv('CORS_ORIGIN_REGEX') or None
app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_origin_regex=origin_regex, allow_credentials=True,
                   allow_methods=['*'], allow_headers=['*'])

# 선택 가능한 메뉴/재료 목록 (프론트 자동완성용) — AI 엔진의 CSV 기준 (월 824 + 군 260)
MENU_NAMES = core.ALL_MENU_NAMES
ING_NAMES = sorted({ig.split(',')[0].strip()
                    for igs in core.menu_ings.values() for ig in igs})

# 이어짓기용 누적 영양 키. 응답의 intake와 같은 키여야 한다.
CONSUMED_KEYS = ('E', 'protein', 'K', 'P', 'Na', 'Na_season')
# 표시용 nutrition 키. 프론트가 intake 대신 이걸 보내는 실수를 422로 막는다.
NUTRITION_KEYS = {'energy', 'potassium', 'phosphorus', 'sodium', 'sodium_total'}


def parse_consumed(raw):
    """앞 끼 응답의 intake를 가공 없이 그대로 받는다. 키가 틀리면 0으로 먹지 말고 에러."""
    if not raw:
        return None
    missing = [k for k in CONSUMED_KEYS if k not in raw]
    if missing:
        hint = ''
        if NUTRITION_KEYS & set(raw):
            hint = " 표시용 'nutrition'이 아니라 응답의 'intake'를 그대로 보내세요."
        raise HTTPException(422, f'consumed에 {missing} 키가 없습니다.{hint} '
                                 f'필요한 키: {list(CONSUMED_KEYS)}')
    try:
        return {k: float(raw[k] or 0) for k in CONSUMED_KEYS}
    except (TypeError, ValueError):
        raise HTTPException(422, 'consumed 값은 모두 숫자여야 합니다.')


def _parse_birthdate(raw: str):
    """'YYYY-MM-DD' 문자열 → date. 형식이 틀리면 422."""
    from datetime import date
    try:
        y, m, d = (int(x) for x in str(raw).strip().split('-'))
        return date(y, m, d)
    except Exception:
        raise HTTPException(422, '생년월일 형식이 올바르지 않습니다 (YYYY-MM-DD).')


def _age_from(b) -> Optional[int]:
    from datetime import date
    today = date.today()
    age = today.year - b.year - ((today.month, today.day) < (b.month, b.day))
    return age if 1 <= age <= 120 else None


# user_profiles.birthdate 는 002_account_recovery.sql 에서 추가되는 컬럼이다.
# 마이그레이션이 적용되지 않은 환경에서도 계정 찾기 기능이 실패하지 않도록,
# 계정 관련 요청이 처음 들어올 때 한 번만 ADD COLUMN IF NOT EXISTS 를 실행해 둔다.
_birthdate_ready = False


def _ensure_birthdate_column(conn):
    global _birthdate_ready
    if _birthdate_ready:
        return
    conn.execute(text('ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS birthdate date'))
    _birthdate_ready = True


def bearer(authorization: Optional[str] = Header(default=None)):
    if not authorization or not authorization.lower().startswith('bearer '):
        raise HTTPException(401, '로그인이 필요합니다.')
    token = authorization.split(' ', 1)[1].strip()
    with db() as conn:
        user = resolve_user(conn, token)
    if not user:
        raise HTTPException(401, '로그인이 만료되었습니다.')
    return dict(user)


# ────────────────────────── 상태 확인 ──────────────────────────
@app.get('/health')
def health():
    """DB 연결 상태와 AI 생성 엔진 상태를 함께 보고한다."""
    try:
        with db() as conn:
            conn.execute(text('select 1'))
        db_info = {'ok': True}
    except Exception as e:
        # 실제 예외 메시지(DB 호스트·계정명 등이 섞여 나올 수 있음)는 서버 로그에만 남기고,
        # 인증 없이 누구나 호출 가능한 /health 응답에는 내려보내지 않는다.
        print(f'/health DB 점검 실패: {e}')
        db_info = {'ok': False}
    return {
        'ok': db_info['ok'],
        'version': '10.0.0',
        'data_source': 'ai-engine',       # /generate는 항상 실제 생성 엔진을 사용
        'engine': 'klue-bert+reinforce',
        'menus': len(MENU_NAMES),
        'db': db_info,
        'auth': 'database' if db_info['ok'] else 'unavailable',
    }


# ────────────────────────── 회원/인증 (Neon DB) ──────────────────────────
@app.post('/auth/signup', status_code=201)
def signup(req: SignupReq):
    username = req.normalized()  # 아이디 형식 검증(영문/숫자/._- 4~30자)
    with db() as conn:
        # DB 컬럼명은 기존 스키마 그대로(email) 재사용하되, 실제로는 '아이디'로 취급한다.
        if conn.execute(text('select 1 from app_users where lower(email)=:e'), {'e': username.lower()}).first():
            raise HTTPException(409, '이미 사용 중인 아이디입니다.')
        uid = str(uuid.uuid4())
        conn.execute(text('insert into app_users(id,email,password_hash,display_name) values(:i,:e,:p,:n)'),
                    {'i': uid, 'e': username, 'p': hash_password(req.password), 'n': req.name.strip()})
        conn.execute(text("insert into user_profiles(user_id,dialysis_type) values(:u,'혈액투석')"), {'u': uid})
        token = issue_session(conn, uid)
    return {'token': token, 'user': {'id': uid, 'username': username, 'name': req.name.strip()}}


@app.post('/auth/login', dependencies=[Depends(rate_limit('login', 10, 300))])
def login(req: LoginReq):
    username = req.username.strip()
    with db() as conn:
        row = conn.execute(text('select id,email,password_hash,display_name,is_active from app_users where lower(email)=:e'),
                           {'e': username.lower()}).mappings().first()
        if not row or not row['is_active'] or not verify_password(req.password, row['password_hash']):
            raise HTTPException(401, '아이디 또는 비밀번호가 올바르지 않습니다.')
        conn.execute(text('update app_users set last_login_at=now() where id=:i'), {'i': row['id']})
        token = issue_session(conn, str(row['id']))
    return {'token': token, 'user': {'id': str(row['id']), 'username': row['email'], 'name': row['display_name']}}


@app.post('/auth/logout', status_code=204)
def logout(authorization: Optional[str] = Header(default=None)):
    if authorization and authorization.lower().startswith('bearer '):
        import hashlib
        h = hashlib.sha256(authorization.split(' ', 1)[1].strip().encode()).hexdigest()
        with db() as conn:
            conn.execute(text('update auth_sessions set revoked_at=now() where token_hash=:h'), {'h': h})


@app.get('/me')
def me(user=Depends(bearer)):
    with db() as conn:
        _ensure_birthdate_column(conn)
        p = conn.execute(text('''select gender,age,height_cm,weight_kg,dialysis_type,birthdate
                                 from user_profiles where user_id=:u'''),
                         {'u': user['id']}).mappings().first()
    profile = dict(p) if p else None
    if profile and profile.get('birthdate'):
        profile['birthdate'] = profile['birthdate'].isoformat()
    return {'user': {'id': str(user['id']), 'username': user['email'], 'name': user['display_name']},
            'profile': profile}


@app.put('/me/profile')
def update_profile(req: ProfileReq, user=Depends(bearer)):
    try:
        age = req.computed_age()
    except ValueError as e:
        raise HTTPException(422, str(e))
    # 생년월일 원본도 보관한다 — 아이디/비밀번호 찾기의 본인 확인 근거로 쓰인다.
    birth = _parse_birthdate(req.birthdate) if req.birthdate else None
    with db() as conn:
        _ensure_birthdate_column(conn)
        conn.execute(text('''insert into user_profiles(user_id,gender,age,height_cm,weight_kg,dialysis_type,birthdate)
        values(:u,:g,:a,:h,:w,:d,:b) on conflict(user_id) do update set gender=excluded.gender,age=excluded.age,
        height_cm=excluded.height_cm,weight_kg=excluded.weight_kg,dialysis_type=excluded.dialysis_type,
        birthdate=coalesce(excluded.birthdate,user_profiles.birthdate),updated_at=now()'''),
        {'u': user['id'], 'g': req.gender, 'a': age, 'h': req.height, 'w': req.weight,
         'd': req.dialysis, 'b': birth})
    return {'ok': True}


# ── 아이디 찾기 / 비밀번호 찾기 ───────────────────────────────────────────────
# 이 서비스는 이메일·문자 발송 수단이 없어서 "재설정 링크를 보낸다" 방식을 쓸 수 없다.
# 대신 가입 시 프로필에 입력한 이름 + 생년월일이 모두 일치할 때만 알려주거나 재설정한다.
# 프로필을 아직 입력하지 않아 생년월일이 없는 계정은 확인 근거가 없으므로 찾을 수 없다.
@app.post('/auth/find-id')
def find_id(req: FindIdReq):
    birth = _parse_birthdate(req.birthdate)
    with db() as conn:
        _ensure_birthdate_column(conn)
        rows = conn.execute(text('''select u.email, u.created_at
            from app_users u join user_profiles p on p.user_id=u.id
            where btrim(lower(u.display_name))=:n and p.birthdate=:b and u.is_active=true
            order by u.created_at'''),
            {'n': req.name.strip().lower(), 'b': birth}).mappings().all()
    if not rows:
        raise HTTPException(404, '입력하신 이름과 생년월일로 가입된 아이디를 찾지 못했습니다.')
    return {'usernames': [r['email'] for r in rows],
            'joined_at': [r['created_at'].date().isoformat() for r in rows]}


@app.post('/auth/reset-password', dependencies=[Depends(rate_limit('reset-password', 5, 900))])
def reset_password(req: ResetPasswordReq):
    birth = _parse_birthdate(req.birthdate)
    username = req.username.strip()
    with db() as conn:
        _ensure_birthdate_column(conn)
        row = conn.execute(text('''select u.id from app_users u join user_profiles p on p.user_id=u.id
            where lower(u.email)=:e and btrim(lower(u.display_name))=:n and p.birthdate=:b and u.is_active=true'''),
            {'e': username.lower(), 'n': req.name.strip().lower(), 'b': birth}).mappings().first()
        if not row:
            raise HTTPException(404, '아이디·이름·생년월일이 모두 일치하는 계정을 찾지 못했습니다.')
        conn.execute(text('update app_users set password_hash=:p, updated_at=now() where id=:i'),
                     {'p': hash_password(req.new_password), 'i': row['id']})
        # 비밀번호가 바뀌었으니 기존에 로그인돼 있던 세션은 모두 끊는다.
        conn.execute(text('update auth_sessions set revoked_at=now() where user_id=:i and revoked_at is null'),
                     {'i': row['id']})
    return {'ok': True}


# ────────────────────────── 메뉴/재료 목록 (AI 엔진 기준) ──────────────────────────
@app.get('/menus')
def menus(q: Optional[str] = None):
    if q:
        ql = q.strip()
        return {'menus': [m for m in MENU_NAMES if ql in m][:200]}
    return {'menus': MENU_NAMES}


@app.get('/ingredients')
def ingredients(q: Optional[str] = None):
    if q:
        ql = q.strip()
        return {'ingredients': [i for i in ING_NAMES if ql in i][:200]}
    return {'ingredients': ING_NAMES}


@app.get('/veg_potassium_tips')
def veg_potassium_tips():
    """채소 조리 시 칼륨 빼는 방법 (단단한/부드러운 채소별). 정적 콘텐츠."""
    import recipe_editor_FOOK as R
    return {'tips': R.VEG_POTASSIUM_TIPS}


@app.get('/menus_by_ingredient')
def menus_by_ingredient(q: str):
    """재료 검색 화면에서 '이 재료가 들어간 메뉴' 목록을 보여주기 위한 역매핑.
    core.menu_ings는 {메뉴명: [재료명, ...]} 구조이므로, 재료명이 부분일치하는 메뉴를 뽑는다."""
    ql = q.strip()
    if not ql:
        return {'menus': []}
    matched = [m for m, igs in core.menu_ings.items()
              if any(ql in ig for ig in igs)]
    return {'menus': sorted(set(matched))[:100]}


# ────────────────────────── 식단 생성 (진짜 AI 엔진) ──────────────────────────
@app.post('/generate')
def generate(req: GenReq):
    """한 끼 생성. consumed(오늘 먹은 누적)를 주면 남은 예산으로 이 끼의 목표를 잡는다."""
    consumed = parse_consumed(req.consumed)
    return core.meal_result(menu=(req.menu or None),
                            ingredient=(req.ingredient or None),
                            weight=req.weight, height=req.height, sex=req.sex,
                            consumed=consumed,
                            meals_left=max(1, min(3, req.meals_left)))


@app.post('/generate_day')
def generate_day(req: DayReq):
    """하루 세 끼를 남은 예산 방식으로 이어서 생성."""
    def clean(lst):
        return [(x or None) for x in lst] if lst else None
    return core.day_result(weight=req.weight, height=req.height, sex=req.sex,
                           menus=clean(req.menus), ingredients=clean(req.ingredients))


@app.post('/recipe', dependencies=[Depends(rate_limit('recipe', 20, 300))])
def recipe(req: RecipeReq):
    """완성 한끼의 특정 메뉴 조리법을 LLM으로 편집(원본기반+투석팁). OPENAI_API_KEY 필요."""
    import recipe_editor_FOOK as R
    try:
        ings = [(str(i[0]), float(i[1])) for i in req.ingredients]
        return {'menu': req.menu,
                'steps': R.edit_recipe(req.menu, ings, model=req.model, source_menu=req.source)}
    except Exception as e:
        return {'menu': req.menu, 'error': str(e)}


@app.post('/tts', dependencies=[Depends(rate_limit('tts', 20, 300))])
def tts(req: TTSReq):
    """조리과정 텍스트를 음성(mp3)으로 변환. OPENAI_API_KEY 필요."""
    import recipe_editor_FOOK as R
    try:
        audio = R.text_to_speech(req.text, voice=req.voice)
        return Response(content=audio, media_type='audio/mpeg')
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post('/chat', dependencies=[Depends(rate_limit('chat', 20, 300))])
def chat(req: ChatReq):
    """투석·콩팥병 영양 질문에 답하는 RAG 챗봇 (대한신장학회 자료 등 근거). OPENAI_API_KEY 필요.
    weight+consumed(오늘 이미 먹은 양, /generate 응답의 intake)를 같이 주면 '오늘 남은 예산' 감안한
    개인화 답변을 준다 — 재료 질문에서만 적용되고, 일반 지식 질문에는 영향 없음."""
    import FOOK_rag_chatbot as C
    consumed = parse_consumed(req.consumed)   # 형식 틀리면 422 (조용히 0으로 처리 안 함, /generate와 동일 원칙)
    try:
        answer, sources = C.answer(req.question, weight=req.weight, consumed=consumed,
                                    meals_left=req.meals_left)
        return {'answer': answer, 'sources': sources}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ────────────────────────── 개인 데이터 (Neon DB) ──────────────────────────
RESOURCE_TABLES = {'meal-records': 'meal_records', 'favorites': 'favorites', 'documents': 'user_documents'}


@app.get('/me/{resource}')
def list_resource(resource: str, user=Depends(bearer)):
    table = RESOURCE_TABLES.get(resource)
    if not table:
        raise HTTPException(404)
    with db() as conn:
        rows = conn.execute(text(f'select id,title,subtitle,payload,created_at from {table} where user_id=:u order by created_at desc'),
                            {'u': user['id']}).mappings().all()
    return {'items': [dict(r) for r in rows]}


@app.post('/me/{resource}', status_code=201)
def save_resource(resource: str, req: SaveReq, user=Depends(bearer)):
    table = RESOURCE_TABLES.get(resource)
    if not table:
        raise HTTPException(404)
    rid = str(uuid.uuid4())
    with db() as conn:
        # payload는 jsonb 컬럼인데, psycopg3는 맨 dict를 자동으로 json 파라미터로
        # 바꿔주지 않는다(Json/Jsonb로 감싸야 함) — 감싸지 않으면 매 저장마다
        # "cannot adapt type 'dict'"로 500이 난다. meal-records/favorites/documents
        # 저장 전부가 이 한 줄을 거치므로 여기서만 감싸면 셋 다 해결된다.
        conn.execute(text(f'insert into {table}(id,user_id,title,subtitle,payload) values(:i,:u,:t,:s,:p)'),
                    {'i': rid, 'u': user['id'], 't': req.title, 's': req.subtitle, 'p': Jsonb(req.payload)})
    return {'id': rid}


@app.delete('/me/{resource}/{item_id}', status_code=204)
def delete_resource(resource: str, item_id: str, user=Depends(bearer)):
    table = RESOURCE_TABLES.get(resource)
    if not table:
        raise HTTPException(404)
    with db() as conn:
        conn.execute(text(f'delete from {table} where id=:i and user_id=:u'), {'i': item_id, 'u': user['id']})


@app.get('/me/cart')
def get_cart(user=Depends(bearer)):
    with db() as conn:
        rows = conn.execute(text('select id,name,amount,unit,checked from shopping_cart_items where user_id=:u order by created_at'),
                            {'u': user['id']}).mappings().all()
    return {'items': [dict(r) for r in rows]}


@app.post('/me/cart', status_code=201)
def add_cart(req: CartReq, user=Depends(bearer)):
    rid = str(uuid.uuid4())
    with db() as conn:
        conn.execute(text('insert into shopping_cart_items(id,user_id,name,amount,unit,checked) values(:i,:u,:n,:a,:un,:c)'),
                    {'i': rid, 'u': user['id'], 'n': req.name, 'a': req.amount, 'un': req.unit, 'c': req.checked})
    return {'id': rid}


@app.patch('/me/cart/{item_id}')
def patch_cart(item_id: str, payload: dict, user=Depends(bearer)):
    with db() as conn:
        conn.execute(text('update shopping_cart_items set checked=coalesce(:c,checked),updated_at=now() where id=:i and user_id=:u'),
                    {'c': payload.get('checked'), 'i': item_id, 'u': user['id']})
    return {'ok': True}


@app.delete('/me/cart/{item_id}', status_code=204)
def delete_cart(item_id: str, user=Depends(bearer)):
    with db() as conn:
        conn.execute(text('delete from shopping_cart_items where id=:i and user_id=:u'), {'i': item_id, 'u': user['id']})
