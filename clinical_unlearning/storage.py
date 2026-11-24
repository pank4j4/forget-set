"""SQLite experiment history and audit trail."""
from __future__ import annotations

from datetime import datetime
import json
import sqlite3

from clinical_unlearning.config import DB_PATH


# ---------------------------------------------------------------------
# Persistent experiment/audit history
# ---------------------------------------------------------------------
def get_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS experiments(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT, method TEXT, forget_patients INTEGER,
        forget_admissions INTEGER, status TEXT, metrics_json TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS audit(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT, event TEXT, details TEXT)""")
    con.commit()
    return con

def audit(event, details):
    con = get_db()
    con.execute(
        "INSERT INTO audit(created_at,event,details) VALUES(?,?,?)",
        (datetime.now().isoformat(timespec="seconds"), event, json.dumps(details, default=str))
    )
    con.commit()
    con.close()

def save_experiment(method, n_patients, n_admissions, status, metrics):
    con = get_db()
    con.execute(
        """INSERT INTO experiments(created_at,method,forget_patients,
           forget_admissions,status,metrics_json) VALUES(?,?,?,?,?,?)""",
        (datetime.now().isoformat(timespec="seconds"), method, int(n_patients),
         int(n_admissions), status, json.dumps(metrics, default=float))
    )
    con.commit()
    con.close()
