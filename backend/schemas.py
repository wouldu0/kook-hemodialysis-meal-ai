# -*- coding: utf-8 -*-
"""
API 요청 스키마(Pydantic). server_FOOK.py에서 쓰지만, 이 파일 자체는 app_core_FOOK(TF
모델 로딩, 수십 초 소요)을 import하지 않는다 — 그래서 검증 로직만 가볍게 테스트할 수 있다.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

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
    # ProfileReq.weight와 같은 범위(20~300kg)를 그대로 재사용한다 — 0/음수/비현실적으로
    # 큰 값이 그대로 흘러들어가는 걸 막는 raw input sanity bound일 뿐, 임상적으로 '정상
    # 체중 범위'를 뜻하지 않는다. height가 같이 오면(정상 가입 유저는 거의 항상 그럼)
    # 실제 영양 목표 계산에는 이 weight 대신 F.standard_weight(height, sex)가 쓰인다 —
    # weight가 목표 산정에 직접 쓰이는 건 height 없이 호출되는 경우(예: 비회원 체험)뿐.
    weight: int = Field(default=60, ge=20, le=300)
    consumed: Optional[dict] = None
    meals_left: int = 3


class DayReq(HeightSexMixin):
    weight: int = Field(default=60, ge=20, le=300)  # ProfileReq.weight와 동일 범위 — 위 GenReq.weight 주석 참고
    menus: Optional[list] = None
    ingredients: Optional[list] = None

    # day_result()는 menus[mi]/ingredients[mi]를 mi=0..2로 바로 인덱싱한다(하루 세 끼 고정).
    # 길이가 1~2(또는 4+)면 IndexError로 500이 나므로, 여기서 미리 422로 막는다.
    # 빈 리스트([])는 server_FOOK.clean()이 이미 None(=지정 안 함, 셋 다 랜덤)으로 취급해온
    # 기존 동작이라 그대로 허용한다 — None과 [] 둘 다 통과, 길이 1·2·4+만 막는다.
    @field_validator('menus', 'ingredients')
    @classmethod
    def _exactly_three(cls, v):
        if v and len(v) != 3:
            raise ValueError('menus/ingredients를 주려면 정확히 3개(아침/점심/저녁)여야 합니다.')
        return v


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
    # 2026-08-14 추가 — "그럼 몇 조각?" 같은 한 턴짜리 음식 후속 질문 지원용. 대화 이력을 서버가
    # 들고 있지 않으므로(stateless), 직전 응답에서 클라이언트가 그대로 돌려받은 canonical 재료명
    # 문자열 하나만 "힌트"로 받는다 — 서버는 이 값을 절대 그대로 신뢰하지 않고 매번 영양DB에
    # 실존하는 항목인지 검증한다(FOOK_rag_chatbot.answer_with_context() 참고). 안 보내도(기존
    # 클라이언트) 기존 동작과 100% 동일 — 하위호환.
    context_food: Optional[str] = None


class SaveReq(BaseModel):
    title: str
    subtitle: Optional[str] = None
    payload: dict = {}


class CartReq(BaseModel):
    name: str
    amount: Optional[float] = None
    unit: str = 'g'
    checked: bool = False
