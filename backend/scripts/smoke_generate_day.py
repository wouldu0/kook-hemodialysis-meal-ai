"""
/generate, /generate_day 스모크 테스트 — 실제 TF 모델을 띄운 서버에 대고 돌린다.

CI에서는 안 돈다(모델 로딩이 무겁고 느림). 배포 전에 수동으로 한 번 돌리는 용도.
gen_batch()가 전역으로 공유하는 TF 모델(gen.encoder/gen.decoder)을 스레드풀에서
동시에 건드리면 'GRUCell' object has no attribute 'kernel' 레이스가 났던 버그의
회귀 테스트 — app_core_FOOK.py의 _GEN_LOCK + 기동 시 워밍업 호출로 고쳤다.

사용법:
  pip install requests   # requirements.txt엔 없음(서버가 아니라 이 스크립트만 쓰는 클라이언트 의존성)
  python scripts/smoke_generate_day.py                          # 로컬(127.0.0.1:8000)
  python scripts/smoke_generate_day.py --base-url https://kook-backend.onrender.com
                                                                  # Render 무료 티어는 콜드스타트로
                                                                  # 첫 응답이 1분 이상 걸릴 수 있다.
"""
import argparse
import concurrent.futures
import sys
import time

import requests

BODY_SINGLE = {"weight": 60, "height": 170, "sex": "남"}
BODY_DAY = {"weight": 60, "height": 170, "sex": "남"}


def call(base_url: str, path: str, body: dict, timeout: float = 90) -> tuple[bool, str]:
    try:
        r = requests.post(f"{base_url}{path}", json=body, timeout=timeout)
        if r.status_code != 200:
            return False, f"{path} -> HTTP {r.status_code}: {r.text[:200]}"
        return True, f"{path} -> 200 OK"
    except Exception as e:
        return False, f"{path} -> EXCEPTION: {e}"


def run_case(name: str, fn) -> bool:
    print(f"\n=== {name} ===")
    t0 = time.time()
    ok = fn()
    dt = time.time() - t0
    print(f"[{'PASS' if ok else 'FAIL'}] {name} ({dt:.1f}s)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--repeat", type=int, default=15, help="/generate_day 연속 호출 횟수")
    ap.add_argument("--concurrency", type=int, default=6, help="동시 요청 스트레스 테스트 동시 개수")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    results: list[bool] = []

    def health():
        try:
            r = requests.get(f"{base}/health", timeout=30)
            print(f"/health -> {r.status_code} {r.text[:200]}")
            return r.status_code == 200
        except Exception as e:
            print(f"/health -> EXCEPTION: {e}")
            return False

    results.append(run_case("0) 서버 헬스체크(콜드스타트 대비 넉넉히 대기)", health))

    def same_input_5x():
        ok = True
        for i in range(5):
            success, msg = call(base, "/generate_day", BODY_DAY)
            print(f"  [{i + 1}/5] {msg}")
            ok = ok and success
        return ok

    results.append(run_case("1) 같은 입력으로 /generate_day 5회 반복", same_input_5x))

    def single_then_day():
        s1, m1 = call(base, "/generate", BODY_SINGLE)
        print(f"  {m1}")
        s2, m2 = call(base, "/generate_day", BODY_DAY)
        print(f"  {m2}")
        return s1 and s2

    results.append(run_case("2) /generate(단일 생성) 후 /generate_day", single_then_day))

    def day_repeat_n():
        ok = True
        for i in range(args.repeat):
            success, msg = call(base, "/generate_day", BODY_DAY)
            print(f"  [{i + 1}/{args.repeat}] {msg}")
            ok = ok and success
        return ok

    results.append(
        run_case(f"3) /generate_day 연속 {args.repeat}회", day_repeat_n)
    )

    def day_then_single():
        s1, m1 = call(base, "/generate_day", BODY_DAY)
        print(f"  {m1}")
        s2, m2 = call(base, "/generate", BODY_SINGLE)
        print(f"  {m2}")
        return s1 and s2

    results.append(run_case("4) 하루 생성 후 다시 한 끼 생성", day_then_single))

    def concurrent_burst():
        """실제 버그의 근본 원인(동시 요청이 전역 TF 모델을 동시에 건드림)을 직접 재현."""
        ok = True
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [
                ex.submit(call, base, "/generate_day", BODY_DAY)
                for _ in range(args.concurrency)
            ]
            for i, f in enumerate(concurrent.futures.as_completed(futs)):
                success, msg = f.result()
                print(f"  [{i + 1}/{args.concurrency}] {msg}")
                ok = ok and success
        return ok

    results.append(
        run_case(
            f"5) /generate_day 동시 {args.concurrency}개 요청(원래 버그 재현 시나리오)",
            concurrent_burst,
        )
    )

    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"결과: {passed}/{total} 케이스 통과")
    if passed != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
