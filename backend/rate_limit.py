# -*- coding: utf-8 -*-
"""
매우 가벼운 인메모리 rate limiter.

Render 무료 플랜은 프로세스 1개로만 도니 이 정도(프로세스 내 dict)면 충분하다.
나중에 여러 인스턴스로 수평 확장하게 되면 Redis 등 공유 저장소 기반으로 바꿔야 한다
(그 전까지는 인스턴스마다 카운트가 따로 논다).

IP별 슬라이딩 윈도우 방식: 최근 window_seconds 안의 요청 시각을 큐에 쌓아두고,
개수가 max_requests를 넘으면 429로 막는다.
"""
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

_LOCK = threading.Lock()
_HITS: dict[str, deque] = defaultdict(deque)
# 오래 떠 있는 서버에서 다녀간 IP가 계속 쌓이는 걸 막기 위한 느슨한 정리 기준.
_CLEANUP_THRESHOLD = 10_000


def _client_key(request: Request) -> str:
    # 프록시(Render) 뒤에 있으므로 X-Forwarded-For를 우선 본다. 여러 홉이면 첫 번째(클라이언트) IP.
    fwd = request.headers.get('x-forwarded-for')
    if fwd:
        return fwd.split(',')[0].strip()
    return request.client.host if request.client else 'unknown'


def rate_limit(key_prefix: str, max_requests: int, window_seconds: int):
    """FastAPI `Depends()`로 붙이는 팩토리.

    예: dependencies=[Depends(rate_limit('login', 10, 300))]  # 5분에 10회
    """

    def _dep(request: Request):
        key = f'{key_prefix}:{_client_key(request)}'
        now = time.time()
        with _LOCK:
            if len(_HITS) > _CLEANUP_THRESHOLD:
                for k in [k for k, dq in _HITS.items() if not dq]:
                    del _HITS[k]
            dq = _HITS[key]
            while dq and now - dq[0] > window_seconds:
                dq.popleft()
            if len(dq) >= max_requests:
                retry_after = max(1, int(window_seconds - (now - dq[0])) + 1)
                raise HTTPException(
                    status_code=429,
                    detail=f'요청이 너무 많습니다. {retry_after}초 후 다시 시도해주세요.',
                    headers={'Retry-After': str(retry_after)},
                )
            dq.append(now)

    return _dep
