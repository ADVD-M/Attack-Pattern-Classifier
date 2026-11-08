# Unseen-attack detection: presenter explainer

## Pipeline walkthrough

The loader verifies exact supplied file hashes, row/column counts, label families, difficulty values and unseen membership. Training has 22 attack names plus normal. The corrected test file came from the jmnwong/NSL-KDD-Dataset mirror; the Zenodo test copy is unsuitable.

Each of five shuffled, stratified training folds fits its own log transforms, constant-column selection, scaler and encoder. Labels, family, seen status and difficulty never enter model inputs. Normal-only PCA, density fits and empirical CDF references use only normal training-fold rows. The external test file is evaluated after full-training fits.

Naive Bayes smoothing is selected by three-fold inner CV with inner-fold preprocessing. GMM component count is selected by BIC on normal training rows. PCA keeps enough components for 95% variance unless the 30-component cap binds; the actual retained fraction is recorded in results/audit.json.

Anomaly scores become empirical normal-traffic percentiles. Probability scores remain probabilities. All scores are rounded to four decimals before threshold selection and export so Python and the dashboard make identical decisions. Hybrid is the maximum of RF and GMM scores.

OOF thresholds minimize (r × FN + FP) / N over every unique score, plus an all-negative endpoint. A connection is flagged when score >= threshold. Ties choose the smallest threshold. Threshold fitting never sees test outcomes. Reported CV operating-point metrics reuse OOF labels for threshold selection, so they are validation estimates, not an additional untouched evaluation.

The cost console uses 200-bin OOF histograms for arbitrary cost ratios. This is a threshold approximation; the saved r=10 metrics use exact OOF thresholds. Score 1 occupies the final histogram bin; threshold 1.0001 means no alerts. The theoretical posterior boundary is 1/(1+r), applicable to calibrated probabilities only; neither model is asserted to be calibrated.

## Three lines per model

- Random Forest: averages 200 decision trees. It models nonlinear interactions with few distribution assumptions. It may miss novel patterns unlike its labeled training attacks.
- Gaussian Naive Bayes: assumes conditional feature independence. Each feature has a class-conditional Gaussian likelihood. One-hot and correlated traffic features violate these assumptions and can produce extreme probabilities.
- Gaussian Mixture: fits a full-covariance mixture after normal-only PCA. Low normal density gives a high anomaly score. Novel attacks that resemble normal traffic may evade it.
- Isolation Forest: isolates normal observations with random partitions. Shorter isolation paths indicate unusual connections. Unusual normal traffic can also raise alarms.
- Hybrid: takes max(RF,GMM) without fitting. One threshold implements the OR of the two component decisions at that same threshold. Its independently optimized threshold can differ from each standalone model.

## Actual results at r = 10

| Model | CV accuracy | Test accuracy | Test recall | Seen recall | Unseen recall | Cost |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 99.77% | 86.62% | 79.81% | 84.03% | 69.60% | 1.1682 |
| Gaussian Naive Bayes | 46.54% | 56.92% | 100.00% | 100.00% | 100.00% | 0.4308 |
| Gaussian Mixture | 88.71% | 88.87% | 88.71% | 92.15% | 80.37% | 0.6898 |
| Isolation Forest | 86.41% | 85.85% | 88.38% | 85.84% | 94.53% | 0.7367 |
| RF + GMM | 97.24% | 82.26% | 71.36% | 83.28% | 42.48% | 1.6449 |

Majority-class test accuracy: 56.92% (always predict attack). This is a comparison baseline, not a guaranteed lower bound on every cost-sensitive operating point.

## Plausibility and interpretation

Cost-sensitive thresholds favor recall when misses cost ten times as much as false alarms. Therefore high-accuracy ranges quoted for ordinary thresholds need not hold here. The train/test gap can reflect several distribution changes; per-name unseen slices identify novelty directly, while the gap alone does not prove novelty caused it. Binary stratification does not guarantee every rare attack name occurs in every training fold, so CV can occasionally hold out all instances of an exceptionally rare name. Test labels are used for reporting only.

## Seed stability

| Model | Setting | Metric | Maximum absolute change vs seed 42 |
|---|---|---|---:|
| rf | cv | accuracy | 0.000540 |
| rf | cv | precision | 0.001320 |
| rf | cv | recall | 0.000171 |
| rf | cv | f1 | 0.000577 |
| rf | cv | fpr | 0.001158 |
| rf | cv | auc | 0.000001 |
| rf | cv | cost | 0.000421 |
| rf | test | accuracy | 0.026348 |
| rf | test | precision | 0.027193 |
| rf | test | recall | 0.024234 |
| rf | test | f1 | 0.025381 |
| rf | test | fpr | 0.029760 |
| rf | test | auc | 0.003665 |
| rf | test | cost | 0.148820 |
| nb | cv | accuracy | 0.000000 |
| nb | cv | precision | 0.000000 |
| nb | cv | recall | 0.000000 |
| nb | cv | f1 | 0.000000 |
| nb | cv | fpr | 0.000000 |
| nb | cv | auc | 0.002682 |
| nb | cv | cost | 0.000000 |
| nb | test | accuracy | 0.000000 |
| nb | test | precision | 0.000000 |
| nb | test | recall | 0.000000 |
| nb | test | f1 | 0.000000 |
| nb | test | fpr | 0.000000 |
| nb | test | auc | 0.000000 |
| nb | test | cost | 0.000000 |
| gmm | cv | accuracy | 0.012884 |
| gmm | cv | precision | 0.021937 |
| gmm | cv | recall | 0.011086 |
| gmm | cv | f1 | 0.010138 |
| gmm | cv | fpr | 0.031154 |
| gmm | cv | auc | 0.003774 |
| gmm | cv | cost | 0.038810 |
| gmm | test | accuracy | 0.011977 |
| gmm | test | precision | 0.033921 |
| gmm | test | recall | 0.029222 |
| gmm | test | f1 | 0.009171 |
| gmm | test | fpr | 0.054577 |
| gmm | test | auc | 0.001846 |
| gmm | test | cost | 0.142832 |
| iforest | cv | accuracy | 0.008256 |
| iforest | cv | precision | 0.011706 |
| iforest | cv | recall | 0.001313 |
| iforest | cv | f1 | 0.006733 |
| iforest | cv | fpr | 0.016587 |
| iforest | cv | auc | 0.001298 |
| iforest | cv | cost | 0.006144 |
| iforest | test | accuracy | 0.008251 |
| iforest | test | precision | 0.015062 |
| iforest | test | recall | 0.036546 |
| iforest | test | f1 | 0.011094 |
| iforest | test | fpr | 0.029142 |
| iforest | test | auc | 0.003864 |
| iforest | test | cost | 0.195484 |
| hybrid | cv | accuracy | 0.002254 |
| hybrid | cv | precision | 0.004762 |
| hybrid | cv | recall | 0.000375 |
| hybrid | cv | f1 | 0.002297 |
| hybrid | cv | fpr | 0.004544 |
| hybrid | cv | auc | 0.000044 |
| hybrid | cv | cost | 0.001349 |
| hybrid | test | accuracy | 0.053806 |
| hybrid | test | precision | 0.008010 |
| hybrid | test | recall | 0.102392 |
| hybrid | test | f1 | 0.061809 |
| hybrid | test | fpr | 0.010401 |
| hybrid | test | auc | 0.004403 |
| hybrid | test | cost | 0.578380 |
| rf | test | seen_recall | 0.013542 |
| rf | test | unseen_recall | 0.066933 |
| nb | test | seen_recall | 0.000000 |
| nb | test | unseen_recall | 0.000000 |
| gmm | test | seen_recall | 0.003083 |
| gmm | test | unseen_recall | 0.107467 |
| iforest | test | seen_recall | 0.004844 |
| iforest | test | unseen_recall | 0.113333 |
| hybrid | test | seen_recall | 0.045029 |
| hybrid | test | unseen_recall | 0.241333 |

Seeds 41 and 43 rerun all folds, model selection, normalization and operating thresholds. Values are absolute metric units (multiply by 100 for percentage points), except cost.

## Observed plausibility flags

Naive Bayes has test FPR 100.00% and recall 100.00% at threshold 0.0000. In this run, the OOF cost optimum is to flag all connections. This explains its low accuracy and low cost: missing no attacks is favored when misses cost ten times as much as false alarms. Its AUC of 0.7790 measures ranking separately from this operating decision. Do not present the low cost as a useful selective triage system.

GMM test accuracy is 88.87% with FPR 10.93%. Its normal-only model can catch attacks outside supervised decision regions. This is a result of this dataset and operating point, not evidence that density models always outperform supervised classifiers.

Hybrid test recall is 71.36% at its independently optimized threshold 0.9550. OR dominance only applies when the component models use the same threshold as the hybrid. Independently selected thresholds explain why this hybrid can have lower recall than either standalone detector. The observed result was retained without tuning toward an expected narrative.

No supervised test accuracy exceeds 95%. Training-only fit provenance and independent OOF threshold recomputation pass. High-90s CV accuracy applies to RF here; the NB all-alert cost policy does not satisfy that heuristic.

## Verification

All seven automated checks passed; see results/validation.txt. Run `python -m unittest discover -s tests -v` to repeat them. The tests check file integrity, disjoint preprocessing and held-out row IDs, normal-only fit IDs and CDF population sizes, training-only thresholds, exclusion of difficulty, and Python/JavaScript confusion and cost agreement at multiple thresholds. The scaler test deliberately refits on train plus test and demonstrates changed parameters; runtime provenance audits establish which rows the actual fits used.

The complete three-seed experiment took 369.3 seconds on the development machine. Hardware and library versions affect runtime. Audit records include PCA variance, GMM convergence, BIC and fitted-row provenance.

## 15 likely viva questions

1. **What is the prediction target?** Normal versus attack; family and name are reporting slices, not classes to train.

2. **How many observations are used?** 125,973 training and 22,544 external test connections.

3. **What makes an attack unseen?** Its name is absent from the full training file: 3,750 test records across 17 names.

4. **How good is RF on unseen attacks?** Its recall is 69.60% versus 84.03% on seen attacks at the saved r=10 threshold.

5. **What is RF test accuracy?** 86.62% at the training-selected cost threshold.

6. **Why not select thresholds on the test set?** That would use evaluation outcomes to tune decisions and make the reported cost optimistic.

7. **What does r=10 mean?** One missed attack costs as much as ten false alarms.

8. **What is the RF cost?** 1.1682 per test connection, computed as (10 FN + FP)/22,544.

9. **Why use a normal-only model?** It can flag deviations without examples of each attack name; novelty does not guarantee a deviation.

10. **Why PCA before GMM?** It reduces correlated high-dimensional features and full-covariance computation; its normal-only fit avoids using held-out observations.

11. **Are anomaly scores probabilities?** No. They are empirical percentiles of fitted normal-traffic scores.

12. **Why retain Naive Bayes?** It is a classical Bayes baseline whose violated Gaussian and independence assumptions help interpret its results.

13. **Can the hybrid always win?** No. OR decisions may improve recall while increasing false alarms; threshold changes also affect the comparison.

14. **Why report attack-name counts?** Some names have only two observations, so one error changes recall by 50 percentage points.

15. **What would you do next?** Connect alerts to a SIEM with cost ratio controlling volume, repeat on UNSW-NB15 or CIC-IDS2017, and explain per-feature anomaly contributions.
