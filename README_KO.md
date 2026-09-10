# 과거 1년간 죽고 싶다는 생각 관련 연구 — 분석 코드

**공개용 코드 저장소 v2.0 (2026-09-10)**

이 저장소는 **“Psychosocial Factors Associated With Past-Year Thoughts of Wanting to Die in Korean Adults”** 원고의 분석 및 결과 재생성 코드를 공개용으로 정리한 것입니다. 최종 분석 표본은 **6,605명**이며, 1,850명(28.0%)이 과거 1년간 죽고 싶다는 생각을 보고했습니다.

분석은 크게 세 축으로 구성됩니다: **(1) 조정 연관성(regression), (2) 예측 기여도(machine learning/SHAP), (3) 조건부 네트워크 연결성(network analysis)**. 세 방법은 서로 다른 질문을 다루며 결과를 하나의 인과효과로 해석하지 않습니다.

## 폴더 구성

- `analysis/01_descriptive/` — Table 1 기술통계 및 P값
- `analysis/02_regression/` — 주분석 회귀, spline, PHQ-8/범주형/성별 상호작용 민감도 분석
- `analysis/03_machine_learning_shap/` — 6개 모델, repeated nested CV, bootstrap, calibration, held-out SHAP
- `analysis/04_network/` — EBICglasso, centrality/bridge EI, bootstrap
- `reporting/` — Figure 3, Supplementary Figure S5-S7, Table S4 생성
- `data_templates/` — 헤더만 있는 입력 템플릿; 실제 자료 없음
- `provenance/` — 최종 설정, 실행 manifest, R 패키지 버전 정보
- `verification/` — 참여자 수준 정보가 없는 핵심 집계 검증 결과
- `docs/` — 실행 가이드, 변수 설명, 원고-코드 대응표

## 데이터 공개 제한

실제 참여자 수준 자료와 참여자별 예측결과/checkpoint는 포함하지 않습니다. 자세한 내용은 `docs/DATA_AVAILABILITY_KO.md`를 참고하세요.

## 실행

한국어 전체 실행 순서는 `docs/RUN_GUIDE_KO.md`에 정리했습니다.

## v2.0 변경 범위

공개 저장소에서 읽기 쉽도록 **폴더 구조, 파일명, 문서, Supplement 표·그림의 표기만 정리**했습니다. 통계모형, 분석값, 결과, 추론적 판단은 변경하지 않았습니다.

현재 별도 소프트웨어 라이선스는 부여하지 않았습니다.
