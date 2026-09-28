# Analytic variable dictionary

| Variable | Role / coding used in analysis |
|---|---|
| `participant_id` | Local participant identifier; not distributed. |
| `py_thoughts_wanting_to_die` | Self-reported past-year thoughts of wanting to die (0 = no, 1 = yes). |
| `age_years` | Age in years; regression reports age per 10-year increase. |
| `sex_female` | 0 = male, 1 = female. |
| `married_current` | 0 = not currently married, 1 = currently married. |
| `marital_status_4cat` | 1 = single, 2 = married, 3 = widowed, 4 = divorced; married is the reference category. |
| `employed_corrected` | 0 = not currently working, 1 = currently working. |
| `living_alone` | 0 = living with others, 1 = living alone. |
| `poor_subjective_physical_health` | Four-level physical-health item; higher values indicate poorer health. |
| `poor_subjective_mental_health` | Four-level mental-health item; higher values indicate poorer health. |
| `gad7_total` | GAD-7 total score; primary regression spline knots are 0, 2, 5, 14. |
| `pss14_total` | PSS-14 total score; regression reports effects per 1 SD. |
| `low_self_esteem_score` | Ten-item score ranging from 10 to 40; higher values indicate lower self-esteem. Regression reports effects per 1 SD. |
| `low_sense_of_belonging` | Reverse-coded four-level item; higher values indicate lower belonging. |
| `low_perceived_social_equality` | Reverse-coded four-level item; higher values indicate lower perceived equality. |
| `low_social_trust` | Reverse-coded four-level item; higher values indicate lower trust. |
| `phq8_total` | PHQ-8 total score, used in the depressive-symptom sensitivity analysis with spline knots 0, 3, 6, 16. |

## Response labels in analytic order

Labels follow the English translations in Supplementary Table S7. These are labels for the existing numeric codes, not additional recoding operations.

| Item | Code 1 | Code 2 | Code 3 | Code 4 |
|---|---|---|---|---|
| Subjective physical and mental health | Very healthy | Fairly healthy | Not healthy | Very poor |
| Sense of belonging | Very much | Much | Little | Very little |
| Perceived social equality | Very equal | Equal | Unequal | Very unequal |
| Social trust | Can be trusted very much | Generally can be trusted | Generally cannot be trusted | Cannot be trusted at all |

The two health items retain their raw 1–4 order. Belonging, equality, and trust use `analytic = 5 - raw response`. The public analysis scripts expect these already-coded columns; do not reverse-code them a second time.

The primary regression uses one-category trends for these five items. The categorical sensitivity analysis uses code 2 as the reference for each. Original Korean questionnaire wording is reported in Supplementary Table S7.
