# KOOK 백엔드 (FastAPI + TensorFlow 식단 생성 엔진)
# Render / Railway / Fly.io / Hugging Face Spaces 어디서든 이 파일 하나로 배포된다.
#
# 로컬 개발 환경이 Python 3.9 + tensorflow-cpu 2.20 조합이라 그대로 맞췄다.
# (requirements.txt의 numpy 2.0.2 / scipy 1.13.1은 3.9~3.12용 휠만 있다)
FROM python:3.9-slim

# TF 로그 억제 + keras 2 호환 모드 (server_FOOK.py가 기대하는 설정과 동일)
ENV TF_CPP_MIN_LOG_LEVEL=3 \
    TF_USE_LEGACY_KERAS=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

# matplotlib/h5py 빌드에 필요한 최소 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 설치해서 코드만 바뀔 때 캐시를 재사용한다 (TF 설치가 5~10분 걸린다)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 앱 코드 + 데이터 + 모델 체크포인트
COPY . .

# 플랫폼마다 주는 포트가 다르다. PORT가 없으면 7860(HF Spaces 기본값)을 쓴다.
ENV PORT=7860
EXPOSE 7860

# 모델 로딩에 수십 초 걸리므로 워커는 1개로 둔다(여러 개면 메모리가 배수로 늘어난다).
CMD uvicorn server_FOOK:app --host 0.0.0.0 --port ${PORT}
