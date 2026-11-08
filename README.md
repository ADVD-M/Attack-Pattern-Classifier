# SIGNAL: unseen-attack detection on NSL-KDD

A reproducible classical pattern-recognition experiment and offline security-operations dashboard.

Most intrusion-detection demos report near-perfect accuracy on attacks the model was trained on. This project tests on attacks the model has never seen, and compares a supervised classifier for known attacks with a normal-only density detector as a tripwire for novel ones. The alert threshold is set by the cost of a missed attack against a false alarm. It is designed as a triage layer for security operations or edge networks where labeled attack data is scarce.

## Open the presentation

Double-click `dashboard/index.html`. All five tabs work offline, including the cost slider, ROC operating point and alert details. No server, build, installation or network is needed to view the generated results.

## Reproduce

Python 3.10+ and Node.js (for cross-language verification) are required.

`requirements.txt` specifies compatible minimum versions. `requirements-lock.txt` records the exact scientific-library versions used for the delivered results; install that file for closer reproduction on Python 3.12. The measured full three-seed run took 369.8 seconds (6 minutes 10 seconds), within the 10-minute limit.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/run_pipeline.py
python -m unittest discover -s tests -v
```

On this workspace an environment is already prepared: `.venv\Scripts\python.exe src\run_pipeline.py` runs directly. The default command runs seed 42 and full stability repetitions at seeds 41 and 43. Stage timings print as it runs. Files in results/ and dashboard/data.js are regenerated; never edit data.js by hand.

## Data and provenance

Use the exact supplied files in `data/`; the original root copies are retained. Training is tab-separated, test is comma-separated. Training has 125,973 records and 22 attack names plus normal. Test has 22,544 records with 3,750 unseen attacks from 17 names. Hash checks guard the supplied versions.

Training source: [Zenodo record](https://zenodo.org/records/17424143). The owner's corrected test file came from the [jmnwong archive mirror](https://github.com/jmnwong/NSL-KDD-Dataset/blob/master/KDDTest%2B.txt), which links its provenance to the original UNB archive. The Zenodo test copy has collapsed labels and corrupted service values and must not be substituted. Cite Tavallaee, Bagheri, Lu and Ghorbani (2009), *A Detailed Analysis of the KDD CUP 99 Data Set*.

## Layout

- `src/load.py`: schema, file integrity, labels and fixed split.
- `src/features.py`: fold-local preprocessing.
- `src/models.py`: RF, Gaussian NB, normal-only GMM/PCA and Isolation Forest; derived hybrid.
- `src/evaluate.py`: exact OOF thresholds, costs, metrics and ROC.
- `src/run_pipeline.py`: three complete seeds and all exports.
- `results/`: metrics, thresholds, slices, stability, fitting audit and complete JSON. NPZ files retain OOF scores for verification.
- `dashboard/`: hand-written HTML/CSS/JavaScript with native SVG charts.
- `tests/`: leakage provenance, file and export integrity, decision boundary and Python/JavaScript agreement.
- `docs/EXPLAINER.md`: generated presenter notes with actual results and 15 viva answers.

## Interpretation

All default metrics use the r=10 threshold fitted on OOF training scores. The cost console uses 200-bin OOF histograms for arbitrary ratios, so its selected threshold can differ slightly from the exact default. Every displayed score uses four-decimal precision consistently with Python. The alert feed displays the first 500 filtered matches; 300 stratified connections have raw features. Score and decision details exist for all records.

Future work: send alerts to a SIEM with cost ratio controlling analyst volume; repeat on UNSW-NB15 or CIC-IDS2017; add per-feature contributions to explain anomaly scores.
