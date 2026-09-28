# 과거 1년간 죽고 싶다는 생각 관련 연구 — 분석 코드

**코드 버전 2.1 (2026-09-28)**

이 저장소는 **“Psychosocial correlates of past-year thoughts of wanting to die in Korean adults across regression, machine learning and network analysis”** 원고의 분석 및 표·그림 생성 코드를 담고 있습니다. 분석 표본은 **6,605명**이며, 1,850명(28.0%)이 과거 1년간 죽고 싶다는 생각을 보고했습니다.

동일한 참여자와 예측변수에 대해 조정 연관성, 교차검증 기반 분류와 SHAP 기여도, 조건부 네트워크 연결성을 비교합니다. 회귀 추론에는 White (HC0) 샌드위치 공분산을 사용합니다. 분류 성능은 미래 자살행동 예측 성능을 의미하지 않으며, SHAP과 네트워크 결과는 인과효과가 아닙니다.

## 폴더 구성

- `analysis/01_descriptive/` — Table 1 기술통계와 P값
- `analysis/02_regression/` — 주 회귀모형, spline, PHQ-8·범주형·성별 상호작용 분석
- `analysis/03_machine_learning_shap/` — 6개 분류모형, 반복 nested CV, bootstrap, calibration, held-out SHAP
- `analysis/04_network/` — EBICglasso, 중심성, bridge expected influence, bootstrap
- `reporting/` — 표·그림 생성 코드
- `data_templates/` — 열 이름만 있는 입력 템플릿
- `environment/`, `provenance/` — 소프트웨어 의존성, 실행 설정, 패키지 정보
- `verification/` — 참여자 수준 정보가 없는 집계 결과
- `tests/` — 합성자료를 이용한 소프트웨어 테스트
- `docs/` — 실행 가이드, 변수 설명, 원고-코드 대응표

## 데이터 공개 제한

IRB 승인 데이터 이용기간이 종료되었으며, 승인 조건과 기관 규정에 따라 참여자 수준 자료의 외부 반출과 공유는 허용되지 않습니다. 승인된 이용기간이 끝나면 해당 자료를 삭제해야 합니다. 참여자 자료, 참여자별 예측확률, 식별자, 분할 인덱스 및 checkpoint는 공개하지 않습니다. 자세한 내용은 `docs/DATA_AVAILABILITY_KO.md`에 있습니다.

## 실행과 소프트웨어

실행 가이드는 `docs/RUN_GUIDE_KO.md`에 있습니다. 입력 파일 구조에 맞는 자료 중 적법한 이용 권한이 있는 자료에만 코드를 적용해야 하며, 원 연구자료는 제공하지 않습니다.

회귀는 Python의 statsmodels를 사용합니다. ML 실행기록의 Python 3.10.9와 패키지 버전은 ML 환경에 해당하며, 이를 별도의 회귀 실행기록으로 취급하지 않습니다. 네트워크 분석은 R 4.5.2를 사용했습니다. 자세한 환경 정보는 `environment/README.md`에 있습니다.

합성자료 테스트:

```bash
python -m unittest discover -s tests -v
```

## 버전 2.1

원고 제목과 응답범주 표기를 제출본에 맞추고, 공분산 인수를 `HC0`로 명시했습니다. 숫자 코딩, 기준범주, spline 구성, ML·네트워크 설정과 기존 집계 결과는 유지했습니다. 변경내역은 `CHANGELOG.md`, 인용 정보는 `CITATION.cff`를 참고하세요.

현재 별도 소프트웨어 라이선스는 부여하지 않았습니다.
