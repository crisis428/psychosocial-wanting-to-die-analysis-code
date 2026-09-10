# 실행 가이드

실제 연구 데이터는 저장소에 포함하지 않습니다. 승인된 로컬 환경에서 `data_templates/`의 열 구조와 동일한 CSV를 준비해 실행합니다.

## 1. Table 1 기술통계

```bash
python analysis/01_descriptive/table1_descriptive_statistics.py --input primary_cc.csv --output table1_statistics.csv
```

평균(SD), No. (%), Welch t test, Pearson 카이제곱 검정을 생성합니다.

## 2. 회귀 및 민감도 분석

```bash
python analysis/02_regression/regression_analysis.py --primary primary_cc.csv --phq8 phq8_cc.csv --categorical categorical_sensitivity.csv --outdir regression_results
```

## 3. 머신러닝 및 SHAP

`analysis/03_machine_learning_shap/machine_learning_nested_cv_shap.py`를 실행합니다. 6개 분류모델을 5-fold × 3-repeat nested stratified cross-validation으로 평가하고 held-out SHAP을 계산합니다. 참여자 수준 정보가 들어 있는 내부 checkpoint ZIP은 공개 저장소에 포함하지 않습니다.

## 4. 네트워크 분석

`analysis/04_network/network_analysis.R`을 `primary_cc.csv`가 있는 작업 폴더에서 실행하거나 `WORK_DIR`을 지정합니다. 빠른 점검은 `TEST_MODE <- TRUE`, 최종 계산은 `FALSE`입니다.

## 5. 표·그림 재생성

- 보충 네트워크 그림: `reporting/figures/supplementary_network_figures.R`
- Table S4: `reporting/tables/table_s4_ml_performance.py`
- Figure 3: `reporting/figures/figure3_network_fixed_layout.py`

Figure 3은 세 패널에 total-sample 좌표를 공통 적용해 시각적 비교를 쉽게 한 것이며, 노드 간 거리 자체에는 통계적 의미가 없습니다.
