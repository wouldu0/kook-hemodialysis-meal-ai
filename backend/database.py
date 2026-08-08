from __future__ import annotations
import os
from contextlib import contextmanager
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL', '').strip()
if DATABASE_URL.startswith('postgresql://'):
    DATABASE_URL = DATABASE_URL.replace('postgresql://', 'postgresql+psycopg://', 1)
# DATABASE_URL이 없어도 이 모듈의 import(→ server_FOOK.py 전체)는 죽지 않아야 한다.
# 실제로 없을 때만, db()를 호출하는 회원 관련 라우트에서만 에러가 나야 한다.
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300) if DATABASE_URL else None

@contextmanager
def db():
    if engine is None:
        raise RuntimeError('DATABASE_URL이 없습니다. backend/.env에 Neon 연결 문자열을 설정하세요.')
    with engine.begin() as conn:
        yield conn
