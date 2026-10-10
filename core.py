"""
Everything the app does, without the web screens.
The Streamlit file (app.py) only draws screens and calls these functions.

Three releases of the system, switched live by the instructor:
  v1.0  first release: no input checking at all
  v1.1  same model, now with input checking (validation)
  v2.0  a "new release" with a hidden bug: its new code converts attendance
        to a fraction (80% -> 0.8) before calling the model, but the model was
        trained on percentages. Nothing crashes; the answers are just wrong.
        Used to show monitoring and rollback.
"""
import sqlite3
import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

import joblib
import pandas as pd

VERSIONS = {
    "v1.0": "First release (no input checking)",
    "v1.1": "Input checking added",
    "v2.0": "New release (contains a bug)",
}
DEFAULT_VERSION = "v1.0"
THRESHOLD = 0.5
LOW_MARK = 20  # a mark of 20 or below is rare in the training data


# ---------------------------------------------------------------- model
def load_model(path="model.joblib"):
    return joblib.load(path)


def load_training(path="training_data.csv"):
    return pd.read_csv(path)


# ---------------------------------------------------------------- checking
def validate(attendance_raw, marks_raw):
    """Return (attendance, marks, problem). problem is None when inputs are fine."""
    a_txt, m_txt = str(attendance_raw).strip(), str(marks_raw).strip()
    if a_txt == "" or m_txt == "":
        return None, None, "Both values are required."
    try:
        a, m = float(a_txt), float(m_txt)
    except ValueError:
        return None, None, "Values must be numbers."
    if not 0 <= a <= 100:
        return None, None, f"Attendance must be between 0 and 100 (got {a:g})."
    if not 0 <= m <= 100:
        return None, None, f"Marks must be between 0 and 100 (got {m:g})."
    return a, m, None


# ---------------------------------------------------------------- prediction
def predict(model, version, attendance_raw, marks_raw):
    """Run one request through the chosen release. Never raises."""
    out = {"version": version, "attendance_raw": str(attendance_raw), "marks_raw": str(marks_raw),
           "attendance": None, "marks": None, "status": "ok", "prob": None, "label": None, "reason": None}

    if version == "v1.0":
        # no checking: just try to turn the text into numbers
        try:
            a, m = float(attendance_raw), float(marks_raw)
        except Exception as e:  # noqa: BLE001 - we want to show the raw crash
            out.update(status="error", reason=f"{type(e).__name__}: {e}")
            return out
    else:
        a, m, problem = validate(attendance_raw, marks_raw)
        if problem:
            out.update(status="rejected", reason=problem)
            return out

    model_a = a
    if version == "v2.0":
        model_a = a / 100  # THE BUG: model was trained on percentages, not fractions
    frame = pd.DataFrame([[model_a, m]], columns=["attendance", "marks"])
    p = float(model.predict_proba(frame)[0, 1])
    out.update(attendance=a, marks=m, prob=p,
               label="Needs review" if p >= THRESHOLD else "On track")
    return out


# ---------------------------------------------------------------- shared log
class Store:
    """One small database shared by every participant using the app."""

    def __init__(self, path="class_log.db"):
        self.lock = threading.Lock()
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.execute("""CREATE TABLE IF NOT EXISTS log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, version TEXT,
            attendance_raw TEXT, marks_raw TEXT, attendance REAL, marks REAL,
            status TEXT, prob REAL, label TEXT, reason TEXT)""")
        self.con.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
        self.con.commit()

    def get_version(self):
        with self.lock:
            r = self.con.execute("SELECT value FROM settings WHERE key='version'").fetchone()
        return r[0] if r else DEFAULT_VERSION

    def set_version(self, version):
        assert version in VERSIONS
        with self.lock:
            self.con.execute("INSERT OR REPLACE INTO settings VALUES ('version', ?)", (version,))
            self.con.commit()

    def add(self, rec, ts=None):
        ts = (ts or datetime.now(IST).replace(tzinfo=None)).isoformat(timespec="seconds")  # Indian time
        with self.lock:
            self.con.execute(
                "INSERT INTO log (ts, version, attendance_raw, marks_raw, attendance, marks, status, prob, label, reason)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ts, rec["version"], rec["attendance_raw"], rec["marks_raw"], rec["attendance"],
                 rec["marks"], rec["status"], rec["prob"], rec["label"], rec["reason"]))
            self.con.commit()

    def frame(self):
        with self.lock:
            df = pd.read_sql_query("SELECT * FROM log ORDER BY id", self.con)
        df["ts"] = pd.to_datetime(df["ts"])
        return df

    def reset(self):
        with self.lock:
            self.con.execute("DELETE FROM log")
            self.con.commit()


# ---------------------------------------------------------------- drift check
def drift_report(log, training, minutes=10):
    """Compare recent marks with the marks the model was trained on.

    Deliberately simple so it can be explained in one sentence:
    'Is the average mark, or the share of very low marks, now very
    different from what the model learned from?'
    """
    ok = log[log["status"] == "ok"]
    # ignore impossible values that release v1.0 let through (e.g. marks of 1000)
    ok = ok[ok["marks"].between(0, 100) & ok["attendance"].between(0, 100)]
    if minutes is not None and len(ok):
        ok = ok[ok["ts"] >= ok["ts"].max() - timedelta(minutes=minutes)]
    train_mean = training["marks"].mean()
    train_low = (training["marks"] <= LOW_MARK).mean()
    rep = {"n_recent": len(ok), "train_mean": train_mean, "train_low_share": train_low,
           "recent_mean": None, "recent_low_share": None, "drift": False, "message": ""}
    if len(ok) < 10:
        rep["message"] = "Not enough recent predictions to judge (need at least 10)."
        return rep
    rep["recent_mean"] = ok["marks"].mean()
    rep["recent_low_share"] = (ok["marks"] <= LOW_MARK).mean()
    mean_gap = abs(rep["recent_mean"] - train_mean)
    low_gap = rep["recent_low_share"] - train_low
    rep["drift"] = mean_gap > 15 or low_gap > 0.25
    rep["message"] = (
        f"Average mark now {rep['recent_mean']:.0f} (model learned on {train_mean:.0f}). "
        f"Marks of {LOW_MARK} or below: {rep['recent_low_share']:.0%} now, {train_low:.0%} in training."
    )
    return rep
