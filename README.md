# Clinical Unlearning Lab

Machine unlearning on MIMIC-IV with SISA (Sharded, Isolated, Sliced, Aggregated) training, a shadow-model
membership inference attack, and an audit that compares unlearning against retraining from scratch.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Place the MIMIC-IV tables in `data/raw/` (only `admissions.csv` is mandatory):
`admissions.csv`, `patients.csv`, `diagnoses_icd.csv`, `prescriptions.csv`, `procedures_icd.csv`.

## Project layout

```
Clinical-Unlearning-Lab/
├── app.py                     Entry point: page setup, sidebar, routing
├── requirements.txt
├── data/raw/                  MIMIC-IV CSV files (not committed)
│
├── clinical_unlearning/       Backend (no Streamlit; usable from a notebook or script)
│   ├── config.py              Paths, constants, prediction tasks
│   ├── data.py                CSV loading, features, patient-level split, preprocessing
│   ├── learners.py            Random forest, neural network and SGD shard learners
│   ├── sisa.py                SISA training, prediction and selective unlearning
│   ├── mia.py                 Shadow-model membership inference attack
│   ├── audit.py               Utility, memorisation gap, verdict checks, report
│   ├── metrics.py             Classification metrics
│   └── storage.py             SQLite experiment history and audit trail
│
├── ui/                        Presentation layer
│   ├── theme.py               Global stylesheet
│   ├── components.py          Page header, empty states, charts, styled tables
│   ├── sidebar.py             Navigation, task selector, workflow tracker
│   └── state.py               Session state and cached data loading
│
└── views/                     One module per page, each exposing render(df)
    ├── patient_records.py
    ├── overview.py
    ├── model_training.py
    ├── sisa_architecture.py
    ├── erasure_requests.py
    ├── mia_simulation.py
    ├── unlearning_audit.py
    └── experiment_history.py
```

## Suggested demo flow

1. **Model Training**: create the split, choose the random forest, train.
2. **Erasure Requests**: confirm a Forget Set (for example a random sample).
3. **MIA Simulation**: run Stage 1 (attack before unlearning).
4. **Unlearning & Audit**: run unlearning and read the verdict.
5. **MIA Simulation**: run Stage 2 (same attack after unlearning).

## Adding a page

Create `views/my_page.py` with `def render(df): ...`, then add it to `VIEWS` in `views/__init__.py`.
