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


# 의료진·영양사에게 별도로 안내받은 개인별 영양 기준. 항목별로 값이 있으면 성별·키
# 기반 자동 산출값(F.day_targets) 대신 이 값을 쓰고, 없는(None) 항목은 그대로 자동
# 산출값을 쓴다 — "일부만 override" 구조. 열량·단백질은 [최소,최대] 범위, 칼륨·인·
# 나트륨(첨가염)은 상한 하나다 — 화면에 이미 보여주는 targets/day_targets 응답과
# 정확히 같은 모양이라 그대로 되돌려 보여줄 수 있다.
class CustomTargets(BaseModel):
    energy: Optional[list[float]] = None
    protein: Optional[list[float]] = None
    potassium: Optional[float] = Field(default=None, gt=0, le=10000)
    phosphorus: Optional[float] = Field(default=None, gt=0, le=10000)
    sodium: Optional[float] = Field(default=None, gt=0, le=10000)

    # bool은 파이썬에서 int의 서브타입이라(True==1) pydantic이 기본적으로 float로 조용히
    # 받아준다 — {"potassium": true}가 "1mg 상한"으로 저장되는 걸 막는다.
    @field_validator('potassium', 'phosphorus', 'sodium', mode='before')
    @classmethod
    def _reject_bool_scalar(cls, v):
        if isinstance(v, bool):
            raise ValueError('숫자여야 합니다.')
        return v

    @field_validator('energy', 'protein', mode='before')
    @classmethod
    def _reject_bool_in_range(cls, v):
        if isinstance(v, list) and any(isinstance(x, bool) for x in v):
            raise ValueError('숫자여야 합니다.')
        return v

    @field_validator('energy', 'protein')
    @classmethod
    def _valid_range(cls, v):
        if v is None:
            return v
        if len(v) != 2:
            raise ValueError('[최소, 최대] 형태로 값 2개를 보내야 합니다.')
        lo, hi = v
        if lo <= 0 or hi <= 0:
            raise ValueError('0보다 큰 값이어야 합니다.')
        if lo > hi:
            raise ValueError('최소값이 최대값보다 클 수 없습니다.')
        if hi > 10000:
            raise ValueError('비정상적으로 큰 값입니다.')
        return [float(lo), float(hi)]


class ProfileReq(BaseModel):
    gender: Optional[str] = None
    # 생년월일(YYYY-MM-DD)을 받아 서버에서 나이를 계산해 DB의 age 컬럼에 저장한다.
    # age를 직접 보내는 것도 허용(하위호환) — 둘 다 오면 birthdate가 우선.
    birthdate: Optional[str] = None
    age: Optional[int] = Field(default=None, ge=1, le=120)
    height: Optional[float] = Field(default=None, ge=50, le=250)
    weight: Optional[float] = Field(default=None, ge=20, le=300)
    dialysis: str = '혈액투석'
    # 항상 프로필 전체를 다시 저장하는 구조라, 여기 안 보내면(None) 저장돼 있던 기존
    # custom_targets도 그대로 지워진다 — 프론트는 이 화면(기본정보 수정)에서도 항상
    # 현재 custom_targets 값을 함께 보내 실수로 지우지 않게 한다(ProfileSetupPage 참고).
    custom_targets: Optional[CustomTargets] = None

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
    # 오늘 이미 나온 메뉴 목록(하루 중복 방지). day_result가 끼니마다 누적해서 넘기는 것과
    # 같은 개념을, 프론트가 오늘 실제로 기록한 식사 이력(raw_menus)으로부터 만들어 보낸다.
    used_today: Optional[list[str]] = None
    # 의료진 안내값으로 자동 산출값을 부분 override — 없으면(None) 기존과 완전히 동일하게
    # 동작한다(하위호환). 프론트는 로그인 시 프로필에서 읽어온 값을 매 생성 요청마다 실어 보낸다.
    custom_targets: Optional[CustomTargets] = None


class DayReq(HeightSexMixin):
    weight: int = Field(default=60, ge=20, le=300)  # ProfileReq.weight와 동일 범위 — 위 GenReq.weight 주석 참고
    menus: Optional[list] = None
    ingredients: Optional[list] = None
    custom_targets: Optional[CustomTargets] = None

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


class DayTargetsReq(HeightSexMixin):
    weight: int = Field(default=60, ge=20, le=300)  # ProfileReq.weight와 동일 범위 — 위 GenReq.weight 주석 참고
    custom_targets: Optional[CustomTargets] = None


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
    # GenReq/ProfileReq.weight와 같은 범위(20~300kg)를 그대로 재사용한다 — 감사에서 지적된
    # 항목: 예전엔 ChatReq만 범위 하한/상한이 없어서 0·음수·비현실적으로 큰 값도 그대로
    # food_lookup_answer()의 남은 예산 계산에 흘러들어갈 수 있었다(비개인화 질문에는 영향 없음,
    # weight+consumed가 둘 다 있을 때만 쓰이는 값이라 기존 정상 요청의 허용 범위는 그대로다).
    weight: Optional[int] = Field(default=None, ge=20, le=300)      # 체중(kg). consumed와 같이 주면 '오늘 남은 예산' 감안한 답변
    consumed: Optional[dict] = None   # 오늘 이미 먹은 누적 (generate 응답의 intake와 동일 형식) — 숫자/음수/NaN 검증은 server_FOOK.parse_consumed()에서 (generate와 공유)
    # 오늘 남은 끼니 수(이번 것 포함) — generate/day_result와 동일하게 1~3끼로 제한한다.
    meals_left: Optional[int] = Field(default=None, ge=1, le=3)  # 있으면 하루 전체가 아니라
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
