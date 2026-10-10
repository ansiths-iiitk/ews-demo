"""
Step 2 of the demo: DEPLOYMENT.
The saved model (model.joblib) is loaded by this separate program and made
available to anyone with the link.

Participant page:  https://<your-app>.streamlit.app
Instructor page:   https://<your-app>.streamlit.app/?view=dashboard
"""
import pandas as pd
import streamlit as st

import core

st.set_page_config(page_title="Student Early-Warning Check", page_icon="📋", layout="centered")


# ---------------------------------------------------------------- shared objects
# Loaded once and shared by every visitor, like a real model service.
@st.cache_resource
def get_model():
    try:
        return core.load_model("model.joblib")
    except Exception:
        # safety net: if the hosting server's library version cannot read the
        # saved file, retrain it there (takes about one second)
        import train_model
        train_model.train()
        return core.load_model("model.joblib")


@st.cache_resource
def get_training():
    return core.load_training("training_data.csv")


@st.cache_resource
def get_store():
    return core.Store("class_log.db")


def dashboard_code():
    try:
        return st.secrets["DASHBOARD_CODE"]
    except Exception:  # no secrets set: fall back to the default
        return "factsh2026"


model, training, store = get_model(), get_training(), get_store()


# ---------------------------------------------------------------- participant page
def participant_page():
    st.title("Student Early-Warning Check")
    st.caption("Teaching demonstration. The model was trained on made-up data. Not for real decisions.")

    with st.form("check", clear_on_submit=False):
        attendance = st.text_input("Attendance (%)")
        marks = st.text_input("Internal marks")
        sent = st.form_submit_button("Predict", use_container_width=True)

    if sent:
        version = store.get_version()
        rec = core.predict(model, version, attendance, marks)
        store.add(rec)

        if rec["status"] == "ok":
            if rec["label"] == "Needs review":
                st.error(f"### {rec['label']}")
            else:
                st.success(f"### {rec['label']}")
            st.write(f"Estimated chance this student needs review: **{rec['prob']:.0%}**")
        elif rec["status"] == "rejected":
            st.warning(f"Request not accepted. {rec['reason']}")
        else:
            st.error("The system crashed while handling this request.")
            st.code(rec["reason"])
        st.caption(f"Answered by release {version}")


# ---------------------------------------------------------------- instructor page
def dashboard_page():
    if not st.session_state.get("ok_code"):
        code = st.text_input("Instructor code", type="password")
        if code and code == dashboard_code():
            st.session_state["ok_code"] = True
            st.rerun()
        elif code:
            st.error("Wrong code.")
        return

    st.title("Monitoring dashboard")

    current = store.get_version()
    names = list(core.VERSIONS)
    chosen = st.radio("Live release", names, index=names.index(current), horizontal=True,
                      format_func=lambda v: f"{v} · {core.VERSIONS[v]}")
    if chosen != current:
        store.set_version(chosen)
        st.toast(f"Switched to {chosen}")

    with st.expander("Reset (clear all requests)"):
        if st.checkbox("Yes, delete every logged request") and st.button("Clear log"):
            store.reset()
            st.toast("Log cleared")

    live_panel()


@st.fragment(run_every=3)
def live_panel():
    df = store.frame()
    if df.empty:
        st.info("Waiting for the first request…")
        return

    last5 = df[df["ts"] >= df["ts"].max() - pd.Timedelta(minutes=5)]
    ok = df[df["status"] == "ok"]
    c = st.columns(4)
    c[0].metric("Requests", len(df), f"{len(last5)} in last 5 min")
    c[1].metric("Crashed", int((df["status"] == "error").sum()))
    c[2].metric("Rejected", int((df["status"] == "rejected").sum()))
    c[3].metric("Flagged 'Needs review'", f"{(ok['label'] == 'Needs review').mean():.0%}" if len(ok) else "–")

    # drift check
    rep = core.drift_report(df, training, minutes=10)
    st.subheader("Are tonight's inputs like the training data?")
    if rep["drift"]:
        st.error("Possible drift. " + rep["message"])
    elif rep["recent_mean"] is not None:
        st.success("No drift detected. " + rep["message"])
    else:
        st.info(rep["message"])

    # marks: training vs tonight
    bins = list(range(0, 101, 10))
    labels = [f"{b}–{b + 10}" for b in bins[:-1]]

    def shares(s):
        cut = pd.cut(s.clip(0, 100), bins=bins, labels=labels, include_lowest=True)
        return cut.value_counts(normalize=True).reindex(labels, fill_value=0)

    hist = pd.DataFrame({"Training data": shares(training["marks"])})
    if len(ok):
        hist["Tonight"] = shares(ok["marks"])
    st.caption("Internal marks: share of students in each band")
    st.bar_chart(hist, stack=False, x_label="Marks band", y_label="Share")

    # flagged share over time
    if len(ok):
        per_min = ok.set_index("ts").resample("1min")["label"].apply(
            lambda s: (s == "Needs review").mean() if len(s) else None).dropna()
        if len(per_min) > 1:
            st.caption("Percent flagged 'Needs review', minute by minute")
            st.line_chart((per_min * 100).round(0).rename("Flagged %"))

    st.subheader("Latest requests")
    show = df.tail(25).iloc[::-1].copy()
    show["time"] = show["ts"].dt.strftime("%H:%M:%S")
    show["chance"] = show["prob"].map(lambda p: f"{p:.0%}" if pd.notna(p) else "")
    st.dataframe(show[["time", "version", "attendance_raw", "marks_raw", "status", "label", "chance", "reason"]],
                 hide_index=True, use_container_width=True)

    st.download_button("Download tonight's data (CSV)", df.to_csv(index=False).encode(),
                       "tonight.csv", "text/csv")


# ---------------------------------------------------------------- routing
if st.query_params.get("view") == "dashboard":
    dashboard_page()
else:
    participant_page()
