# Drift detection demo

Reference = validation window (most recent pre-deployment data). PSI warn >= 0.1, alert >= 0.25; dataset drift = >= 1 feature(s) at alert level or score PSI alert. Calendar features excluded. Clean windows = untouched future traffic.

| scenario | n | dataset_drift | share_features_drifted | max_feature_psi | top_drifted | score_psi | score_ks | pr_auc | recall | precision | alerts_per_10k | fp_cost | total_cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean window 1 | 10000 | False | 0.000 | 0.008 | V4=0.01, V10=0.01, V22=0.01 | 0.023 | 0.069 | 0.942 | 0.867 | 0.929 | 14.000 | 15.000 | 120.220 |
| clean window 2 | 10000 | False | 0.000 | 0.029 | log_amount=0.03, V25=0.02, V21=0.01 | 0.106 | 0.138 | 0.845 | 0.722 | 1.000 | 13.000 | 0.000 | 756.070 |
| clean window 3 | 10000 | False | 0.000 | 0.045 | log_amount=0.04, V21=0.03, V25=0.02 | 0.030 | 0.075 | 0.808 | 0.700 | 1.000 | 14.000 | 0.000 | 1462.650 |
| amount drift x0.25 | 30000 | True | 0.034 | 0.465 | log_amount=0.46, V21=0.01, V25=0.01 | 0.043 | 0.088 | 0.858 | 0.755 | 0.976 | 13.667 | 15.000 | 3419.604 |
| amount drift x0.5 | 30000 | True | 0.034 | 0.789 | log_amount=0.79, V21=0.01, V25=0.01 | 0.050 | 0.095 | 0.855 | 0.755 | 0.976 | 13.667 | 15.000 | 4500.268 |
| amount drift x1.0 | 30000 | True | 0.034 | 1.392 | log_amount=1.39, V21=0.01, V25=0.01 | 0.067 | 0.109 | 0.850 | 0.755 | 0.976 | 13.667 | 15.000 | 6661.596 |
| pca drift x0.25 | 30000 | True | 0.034 | 1.339 | V1=1.34, V4=0.23, V7=0.18 | 0.010 | 0.037 | 0.863 | 0.755 | 0.976 | 13.667 | 15.000 | 2338.940 |
| pca drift x0.5 | 30000 | True | 0.207 | 2.133 | V1=2.13, V2=0.49, V7=0.48 | 0.018 | 0.049 | 0.861 | 0.755 | 0.976 | 13.667 | 15.000 | 2338.940 |
| pca drift x1.0 | 30000 | True | 0.207 | 1.890 | V1=1.89, V2=1.87, V7=1.64 | 0.171 | 0.168 | 0.854 | 0.717 | 0.974 | 13.000 | 15.000 | 2369.230 |
| both drift x0.25 | 30000 | True | 0.069 | 1.339 | V1=1.34, log_amount=0.46, V4=0.23 | 0.009 | 0.037 | 0.863 | 0.755 | 0.976 | 13.667 | 15.000 | 3419.604 |
| both drift x0.5 | 30000 | True | 0.241 | 2.133 | V1=2.13, log_amount=0.79, V2=0.49 | 0.017 | 0.046 | 0.860 | 0.755 | 0.976 | 13.667 | 15.000 | 4500.268 |
| both drift x1.0 | 30000 | True | 0.241 | 1.890 | V1=1.89, V2=1.87, V7=1.64 | 0.137 | 0.140 | 0.852 | 0.717 | 0.974 | 13.000 | 15.000 | 6781.652 |
