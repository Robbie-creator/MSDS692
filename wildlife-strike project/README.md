# Rabin Acharya FAA Wildlife Strike Project

Predict reported damaging airport-months at 53 selected airports. This is separate from the Colorado bridge project.

## Reproduce
Use Python 3.12, install requirements.txt, and run from this folder:

    python code/run_analysis.py --input '/path/to/Public(1).xlsx' --output .
    python code/extra_diagnostics.py

The original workbook is supplied separately. selected_source_fields.csv is a compact extraction of the required input fields. Delete that cache when changing source workbooks. A fresh run reads the workbook once. The source audit records its SHA-256 checksum.

## Files
- code/run_analysis.py: source audit, EDA, panel, training-only rolling tuning, chronological validation/test, baseline comparison, probabilities, coefficients, models, plots.
- code/extra_diagnostics.py: precision-recall figure and review-capacity sensitivity.
- results/frozen_specification.json: airport list, feature names, final settings, and selection rule.
- results/model_metrics.csv: 2023 and 2024 performance.
- results/predictions.csv: predictions and monthly review selection flags.
- results/airport_month_panel.csv: complete modeling table.
- models/: serialized scikit-learn models. Load only trusted files.
- figures/: analysis charts.
- submission/: report, ten-slide main deck plus four appendix slides, final report, and talk/Q&A guide.

## Key result
2024 has 636 airport-months and 186 damaging months. Review 11 airports every month (132 reviews): logistic captures 69, forest 67, and airport-history baseline 61. Logistic precision is 52.3% and recall 37.1%; it misses 117 damaging months. This supports a bounded monitored pilot, not demonstrated damage prevention.

## Design
Train 2013–2022 after using 2012 as history. Tune on rolling held-out years 2020, 2021, 2022. Evaluate 2023 and reserved 2024 without refitting on 2023. Training-only airport eligibility requires at least 119 reporting months of 132 and 12 damaging months during 2012–2022, excluding ZZZZ. Lag strike and damage sums by one month. The previous model-switch rule selects forest only for >=5 extra test captures and Brier <= logistic Brier + 0.01. The rule uses the test for final selection, so that selected model requires independent prospective assessment.

## Corrections and limitations
Actual source has 353,804 data rows, eight duplicate identifiers, and 353,796 unique reports. The EDA includes 312,726 and excludes 41,070. Earlier reports had arithmetic inconsistencies. Earlier executable code was unavailable; this implementation transparently reconstructs scope with documented features and settings. Its monthly capacity differs from earlier pooled annual top-20% selection, so scores are not exact replications.
Incident dates do not establish report availability. This retrospective result assumes prior incident reports were available before the forecast month. No flight denominator, causal intervention effect, or defensible savings estimate is available. Descriptive size EDA uses all dates in the supplied extract, while forecasting restricts dates to 2012–2024. Selected airports do not represent all airports. Do not reduce routine prevention on the strength of a low ranking.

## Week 7 use
Submit the status report with its LLM entry. Present main slides 1–10, retain appendix 11–14 for questions. Record your own one-take rehearsal and write observed self-assessment afterward. Actual speaking duration has not been established. The official course status-report template was not supplied; the report follows the attached weekly section structure.
