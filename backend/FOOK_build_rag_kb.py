# -*- coding: utf-8 -*-
"""
FOOK_build_rag_kb.py — 투석/콩팥병 지식베이스 구축 (RAG용)
FOOK_지식베이스_원본/*.txt 를 정제→청킹→임베딩해서 data/FOOK_rag_kb.json 에 저장.

실행: conda activate foodbert; cd /d E:\\final; set OPENAI_API_KEY=sk-...
      python FOOK_build_rag_kb.py
"""
import os, re, json, glob

SRC_DIR = 'FOOK_지식베이스_원본'
OUT = 'data/FOOK_rag_kb.json'
CHUNK_SIZE = 450      # 글자 수 기준 (한국어 문단 단위로 자연스럽게 끊기도록 아래서 문단 경계 우선)
CHUNK_OVERLAP = 80

NOISE_PATTERNS = [
    r'Free!!! 200\+ Paged Kidney Book in 35\+ Languages\s*',
    r'\s*Visit:\s*www\.KidneyEducation\.com\s*',
    r'٢٢',   # 원본 PDF 불릿 글리프가 깨져서 추출된 것
]


def clean_text(t):
    for pat in NOISE_PATTERNS:
        t = re.sub(pat, ' ', t)
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()


def split_paragraphs(t):
    paras = [p.strip() for p in re.split(r'\n\s*\n', t) if p.strip()]
    return paras


MAX_HARD_CHARS = 3000   # 이보다 긴 단일 '문단'(표·목차 등 줄바꿈 없는 구간)은 강제로 잘라 임베딩 토큰 한도 방지


def _hard_split(p, size=CHUNK_SIZE):
    return [p[i:i + size] for i in range(0, len(p), size)]


def chunk_paragraphs(paras, source):
    """문단을 이어붙이다가 CHUNK_SIZE 넘으면 자르고, 다음 청크는 앞 청크 끝부분을 겹쳐서(overlap) 시작.
    문맥이 청크 경계에서 잘려도 이웃 청크에 이어지도록. 단일 문단 자체가 비정상적으로 길면(표·목차 등
    줄바꿈이 없는 구간) 강제로 고정 길이 분할해 임베딩 토큰 한도(8192) 초과를 막는다."""
    chunks = []
    buf = ''
    for p in paras:
        if len(p) > MAX_HARD_CHARS:
            if buf:
                chunks.append(buf); buf = ''
            chunks.extend(_hard_split(p))
            continue
        if len(p) < 15:     # 제목 한 줄짜리 등 너무 짧은 문단은 다음 문단과 합쳐서 버림 방지
            buf += ('\n' if buf else '') + p
            continue
        if len(buf) + len(p) + 1 <= CHUNK_SIZE:
            buf += ('\n' if buf else '') + p
        else:
            if buf:
                chunks.append(buf)
            # overlap: 이전 청크의 뒷부분을 새 청크 앞에 이어붙임
            tail = buf[-CHUNK_OVERLAP:] if buf else ''
            buf = (tail + '\n' + p) if tail else p
    if buf:
        chunks.append(buf)
    return [{'text': c, 'source': source} for c in chunks if len(c.strip()) > 20]


def embed_batch(client, texts, model='text-embedding-3-small'):
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def main():
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise SystemExit('OPENAI_API_KEY 환경변수가 없습니다.')
    from openai import OpenAI
    client = OpenAI(api_key=key)

    all_chunks = []
    for path in sorted(glob.glob(os.path.join(SRC_DIR, '*.txt'))):
        source = os.path.splitext(os.path.basename(path))[0]
        raw = open(path, encoding='utf-8').read()
        cleaned = clean_text(raw)
        paras = split_paragraphs(cleaned)
        chunks = chunk_paragraphs(paras, source)
        print(f'{source}: 문단 {len(paras)}개 -> 청크 {len(chunks)}개')
        all_chunks.extend(chunks)

    print(f'\n총 청크 수: {len(all_chunks)}')
    print('임베딩 생성 중...')

    BATCH = 100
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i:i + BATCH]
        vecs = embed_batch(client, [c['text'] for c in batch])
        for c, v in zip(batch, vecs):
            c['embedding'] = v
        print(f'  {min(i + BATCH, len(all_chunks))}/{len(all_chunks)}')

    os.makedirs('data', exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    print(f'\n저장 완료: {OUT} ({len(all_chunks)}개 청크)')


if __name__ == '__main__':
    main()
