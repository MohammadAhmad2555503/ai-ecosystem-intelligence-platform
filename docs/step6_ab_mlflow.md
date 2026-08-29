# Step 6: A/B Statistical Analysis and MLflow Tracking

Step 6 turns the Step 5 router logs into a statistically defensible experiment result.

## Language

Python 3.11

## Main capability

- A/B test analysis
- Two-proportion z-test
- Power/sample-size calculation
- Confidence interval
- Effect size
- MLflow experiment tracking with local JSON fallback

## Added files

```text
src/evaluation/ab_statistics.py
src/evaluation/ab_log_loader.py
src/evaluation/mlflow_tracker.py
scripts/analyse_ab_test.py
docker/mlflow/requirements.txt
docs/step6_ab_mlflow.md
```

## Primary metric

```text
positive_feedback_rate = percentage of answers rated 4 or 5
```

## Run tests

```bash
python scripts/analyse_ab_test.py --run-tests
```

## Run analysis

```bash
python scripts/analyse_ab_test.py
```

## Outputs

```text
data/evaluation/ab_test_results.json
reports/ab_test_decision_report.md
data/mlflow/ab_test_tracking_fallback.json
```

## Why this matters

This makes the A/B testing serious. The project no longer just splits traffic; it can prove whether the challenger model is statistically better.

