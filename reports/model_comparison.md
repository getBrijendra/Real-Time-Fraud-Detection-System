# Model comparison (test window = last 20% of time, never seen in training or threshold tuning)

Champion (best validation PR-AUC within the 10 ms p50 single-transaction latency budget): **xgboost**

Cost model: review $5/alert, false-positive friction $10, missed fraud = amount + $15 chargeback fee. Thresholds minimise validation cost.

| model | imbalance | val_pr_auc | pr_auc | roc_auc | recall_at_fpr_0.001 | recall_at_fpr_0.005 | recall_at_fpr_0.01 | threshold | precision | recall | fp | fn | cost_false_positives | total_cost | savings_vs_no_model | latency_p50_ms | train_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost | scale_pos_weight | 0.7776 | 0.8118 | 0.9830 | 0.8000 | 0.8800 | 0.8800 | 0.809536 | 0.8889 | 0.7467 | 7 | 19 | $105 | $3,309 | $5,545 | 0.3850 | 38.4 |
| xgboost_unweighted | none | 0.7717 | 0.8000 | 0.9856 | 0.8000 | 0.8400 | 0.8800 | 0.473252 | 0.8871 | 0.7333 | 7 | 20 | $105 | $4,324 | $4,530 | 0.3338 | 14.9 |
| random_forest | class_weight | 0.7616 | 0.8215 | 0.9656 | 0.8267 | 0.8800 | 0.8800 | 0.259874 | 0.8769 | 0.7600 | 8 | 18 | $120 | $3,313 | $5,541 | 28.9 | 91.2 |
| logreg_balanced | class_weight | 0.7559 | 0.7473 | 0.9815 | 0.8000 | 0.8533 | 0.8800 | 0.999994 | 0.9600 | 0.6400 | 2 | 27 | $30 | $3,746 | $5,109 | 1.7057 | 0.6627 |
| logreg_smote | smote | 0.7061 | 0.7403 | 0.9776 | 0.7600 | 0.8533 | 0.8667 | 0.926686 | 0.6395 | 0.7333 | 31 | 20 | $465 | $4,041 | $4,813 | 2.2202 | 1.0858 |
| logreg_undersample | undersample | 0.7045 | 0.7078 | 0.9840 | 0.7467 | 0.8667 | 0.8667 | 0.951077 | 0.7500 | 0.6400 | 16 | 27 | $240 | $4,943 | $3,912 | 1.3837 | 0.1599 |
| logreg_unweighted | none | 0.6077 | 0.6975 | 0.9739 | 0.7600 | 0.8667 | 0.8667 | 0.136094 | 0.6207 | 0.7200 | 33 | 21 | $495 | $4,079 | $4,775 | 1.3080 | 0.5185 |
| isolation_forest | none | 0.0208 | 0.0377 | 0.9483 | 0.0000 | 0.1333 | 0.4667 | 1 | 0.0000 | 0.0000 | 0 | 75 | $0 | $8,854 | $0 | 34.0 | 6.6328 |
