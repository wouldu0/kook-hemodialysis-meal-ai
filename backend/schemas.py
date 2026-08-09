# -*- coding: utf-8 -*-
"""
API 요청 스키마(Pydantic). server_FOOK.py에서 쓰지만, 이 파일 자체는 app_core_FOOK(TF
모델 로딩, 수십 초 소요)을 import하지 않는다 — 그래서 검증 로직만 가볍게 테스트할 수 있다.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

USERNAME_RE = re.compile(r'^[a-zA-Z0-9_.\-]{4,30}$')


def _validate_username(v: str) -> str:
    v = v.strip()
    if not USERNAME_RE.match(v):
        raise ValueError('아이디는 영문/숫자/._- 조합 4~30자여야 합니다.')
    return v


class SignupReq(BaseModel):
    # 이메일 형식이 아니어도 되는 '아이디'. DB의 email 컬럼을 그대로 재사용하되
    # 형식 검증(EmailStr)만 뺐다 — 마이그레이션 없이 기존 스키마 그대로 쓰기 위함.
    username: str = Field(min_length=4, max_length=30)
    password: str = Field(min_length=8, max_length=128)
    name: str = Field(min_length=1, max_length=60)

    def normalized(self) -> str:
        return _validate_username(self.username)


class LoginReq(BaseModel):
    username: str
    password: str


class FindIdReq(BaseModel):
    """아이디 찾기 — 이메일 발송 수단이 없으므로 이름+생년월일로 본인 확인한다."""
    name: str = Field(min_length=1, max_length=60)
    birthdate: str


class ResetPasswordReq(BaseModel):
    """비밀번호 찾기 — 아이디+이름+생년월일이 모두 맞으면 새 비밀번호로 바로 재설정한다."""
    username: str
    name: str = Field(min_length=1, max_length=60)
    birthdate: str
    new_password: str = Field(min_length=8, max_length=128)


class ProfileReq(BaseModel):
    gender: Optional[str] = None
    # 생년월일(YYYY-MM-DD)을 받아 서버에서 나이를 계산해 DB의 age 컬럼에 저장한다.
    # age를 직접 보내는 것도 허용(하위호환) — 둘 다 오면 birthdate가 우선.
    birthdate: Optional[str] = None
    age: Optional[int] = Field(default=None, ge=1, le=120)
    height: Optional[float] = Field(default=None, ge=50, le=250)
    weight: Optional[float] = Field(default=None, ge=20, le=300)
    dialysis: str = '혈액투석'

    def computed_age(self) -> Optional[int]:
        if self.birthdate:
            from datetime import date
            try:
                y, m, d = (int(x) for x in self.birthdate.split('-'))
                b = date(y, m, d)
                today = date.today()
                age = today.year - b.year - ((today.month, today.day) < (b.month, b.day))
                if not (1 <= age <= 120):
                    raise ValueError
                return age
            except Exception:
                # 순수 검증 로직이라 여기선 HTTPException 대신 ValueError로 알리고,
                # 라우트 핸들러(server_FOOK.py)가 FastAPI 응답 형식으로 바꾼다.
                raise ValueError('생년월일 형식이 올바르지 않습니다 (YYYY-MM-DD).')
        return self.age


# F.standard_weight()는 sex가 '여' 계열이면 21, 그 외(None·오타 포함)는 전부 22(남성)로
# 조용히 처리한다. height만 주고 sex를 빠뜨리거나 오타를 내면 성별이 잘못 가정된 채로
# 표준체중·영양 목표가 계산될 수 있어, API 단에서 값 자체와 height 동반 여부를 막는다.
_SEX_VALUES = ('남', '여', 'male', 'female')


class HeightSexMixin(BaseModel):
    height: Optional[float] = Field(default=None, ge=50, le=250)  # 키(cm)
    sex: Optional[Literal['남', '여', 'male', 'female']] = None

    @model_validator(mode='after')
    def _height_requires_sex(self):
        if self.height is not None and self.sex is None:
            raise ValueError(
                f"height를 주려면 sex도 함께 보내야 합니다 ({', '.join(_SEX_VALUES)} 중 하나). "
                "표준체중 계산이 성별에 따라 달라지므로, 빠지면 남성 기준으로 잘못 계산될 수 있습니다."
            )
        return self


class GenReq(HeightSexMixin):
    menu: Optional[str] = None
    ingredient: Optional[str] = None
    weight: int = 60
    consumed: Optional[dict] = None
    meals_left: int = 3


class DayReq(HeightSexMixin):
    weight: int = 60
    menus: Optional[list] = None
    ingredients: Optional[list] = None


class RecipeReq(BaseModel):
    menu: str
    ingredients: list
    model: str = 'gpt-4o-mini'
    source: Optional[str] = None


class TTSReq(BaseModel):
    text: str
    voice: str = 'nova'


class ChatReq(BaseModel):
    question: str
    weight: Optional[int] = None      # 체중(kg). consumed와 같이 주면 '오늘 남은 예산' 감안한 답변
    consumed: Optional[dict] = None   # 오늘 이미 먹은 누적 (generate 응답의 intake와 동일 형식)
    meals_left: Optional[int] = None  # 오늘 남은 끼니 수(이번 것 포함). 있으면 하루 전체가 아니라
                                       # '다음 한 끼 몫'(남은예산÷남은끼니)으로 비교 — generate와 동일 개념


class SaveReq(BaseModel):
    title: str
    subtitle: Optional[str] = None
    payload: dict = {}


class CartReq(BaseModel):
    name: str
    amount: Optional[float] = None
    unit: str = 'g'
    checked: bool = False
