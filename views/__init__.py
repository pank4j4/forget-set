"""One module per workspace page. Each exposes render(df). The order below is the sidebar order."""
from views import (
    patient_records, overview, model_training, sisa_architecture,
    erasure_requests, mia_simulation, unlearning_audit, experiment_history,
)

VIEWS = {
    "Patient Records": patient_records.render,
    "Overview Dashboard": overview.render,
    "Model Training": model_training.render,
    "SISA Architecture": sisa_architecture.render,
    "Erasure Requests": erasure_requests.render,
    "MIA Simulation": mia_simulation.render,
    "Unlearning & Audit": unlearning_audit.render,
    "Experiment History": experiment_history.render,
}
