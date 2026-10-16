# Reporting template — directional claims from seasonal monitoring records

Filled example rows use the values from this package (Poyang and Taihu).
Copy the table, replace the bracketed entries with your own.

| Item | What to report | This package's example (Poyang WL→TN) | This package's example (Taihu TP→Chla) |
|---|---|---|---|
| 1. Variable definition | Species codes used; total-only rerun result | TP/TOTP total-only; mixed-code share reported separately (Section 3.3) | TN/TP/Chla as measured, no composite codes |
| 2. Calendar construction | Gap-length distribution; count of gap-free embedding windows | 84 complete-year months; no interpolated months in the strict layer | 367–370 complete-year months by pair |
| 3. Estimator calibration | MI on independent Gaussian data (nats); k; distance norm | corrected KSG: +0.006 nats (k = 3, max-norm) | same estimator |
| 4. Seasonal null | Test the directional claim against surrogates that preserve seasonal structure; report the surrogate family and draw count | year-block permutation over whole-year blocks (seasonal structure preserved by construction) | same design; 1,000 draws × 3 seeds (42/43/44) |
| 5. Null design & floor | Number of distinct surrogate alignments; the (b+1)/(N+1) floor stated with the p value | 5,040 exact orders incl. identity → floor 2/5,041 ≈ 0.0004 | floor 1/1,001 ≈ 0.001 (11-shift layers: 1/12) |
| 6. Power | Detectable deviation at your power standard; clustered? | 80% power for shares ≥ 58% (n = 316, unclustered) | — |

Verification rules: item 4's surrogates preserve the seasonal structure (year blocks move whole); item 6 uses the clustered bootstrap when stations share gauges (this package: 248 flow stations, 2,000 resamples).
