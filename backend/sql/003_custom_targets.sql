-- 003_custom_targets.sql — 개인별 영양 기준 override 컬럼 추가
--
-- 지금까지는 성별+키로 계산한 표준체중 기반 자동 산출값 하나로 모든 사용자의 영양 기준을
-- 정했다. 하지만 혈액투석 환자는 혈액검사 전해질 수치에 따라 의료진·영양사에게 별도로
-- 안내받은 개인 기준(칼륨/인/나트륨 상한, 열량/단백질 범위)이 있을 수 있다.
-- 이 마이그레이션은 프로필에 그 개인 기준을 저장할 자리(jsonb, 항목별로 값이 있으면
-- override·없으면 기존 자동 산출값 사용)를 추가한다.
--
-- 서버(server_FOOK.py)는 기동 후 첫 프로필 관련 요청 때 이 ALTER를 자동으로 한 번 실행하므로
-- 보통은 이 파일을 직접 돌릴 필요가 없다.

ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS custom_targets jsonb;
