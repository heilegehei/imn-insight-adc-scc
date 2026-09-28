"""IMN Insight product-style interface for the existing ADC vs SCC model."""
from __future__ import annotations

import base64
import hashlib
import html
from pathlib import Path

import streamlit as st
from model_runtime import (
    COMPOSITES, COMPOSITE_INPUTS, DIRECT_COLUMNS, GROUP_LABELS, INPUT_FIELDS,
    LAYERS, LAYER_LABELS, MAIN_LAYER, MODEL_SHA256, SELECTED_DIRECT, THRESHOLDS,
    InputError, example_frame, export_csv, input_template, read_csv_bytes, safe_result_table, score_frame,
)
import pandas as pd

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="IMN Insight | Adenocarcinoma vs Squamous", layout="wide", initial_sidebar_state="collapsed")
st.markdown(f"<style>{(ROOT / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def intro(eyebrow, title, copy=""):
    st.markdown(f'<div class="tile-intro"><div class="tile-eyebrow">{eyebrow}</div><h2>{title}</h2><p>{copy}</p></div>', unsafe_allow_html=True)


def reset_inputs(example=False):
    for f in INPUT_FIELDS:
        value = f["example"] if example else None
        st.session_state[f"input_{f['name']}"] = str(value) if f["name"] == "CRP" and value is not None else ("" if f["name"] == "CRP" else value)
    st.session_state["example_active"] = example
    for key in ("single_result", "single_snapshot"):
        st.session_state.pop(key, None)


def render_hero():
    domain_visual = base64.b64encode((ROOT / "assets" / "imn_domains.svg").read_bytes()).decode("ascii")
    st.markdown(f'''
    <nav class="globalnav"><div class="globalnav-inner"><span class="nav-brand" aria-label="IMN Insight">IMN</span>
    <div class="nav-links"><a href="#overview">Overview</a><a href="#assessment">Assessment</a><a href="#model">Model</a><a href="#batch">Batch</a></div>
    <span class="nav-model">Top-2 Stacking · Three IMN layers</span></div></nav>
    <div class="promo"><b>27 laboratory inputs.</b>&nbsp; Six derived indices. One saved model pathway.</div>
    <section class="hero"><div class="hero-copy"><div class="product-kicker">IMN Insight</div>
    <h1>Adenocarcinoma.<br>Or squamous.</h1>
    <p class="hero-sub">Laboratory signals.<br>Three perspectives. One focused assessment.</p>
    <div class="hero-actions"><a href="#assessment">Start assessment ›</a><a href="#overview">Explore the model ›</a></div></div>
    <div class="hero-stage"><div class="imn-visual">
    <img class="imn-domain-art" src="data:image/svg+xml;base64,{domain_visual}" alt="Connected inflammatory cell, antibody shield, and nutrient leaves representing the three IMN domains.">
    <div class="imn-domain-labels"><div><span class="domain-letter inflammation">I</span><span>Inflammation</span></div><div><span class="domain-letter immunity">M</span><span>Immunity</span></div><div><span class="domain-letter nutrition">N</span><span>Nutrition</span></div></div></div>
    <div class="stage-spec">13 final variables &nbsp; · &nbsp; 2 base learners &nbsp; · &nbsp; 3 IMN layers</div></div></section>
    <section id="overview" class="content-tile"><div class="tile-intro"><div class="tile-eyebrow">The model, at a glance</div>
    <h2>A clear view of the full picture.</h2><p>Inflammation, immunity, and nutrition / metabolism. Enter the laboratory measurements and compare the three saved Stacking outputs.</p></div>
    <div class="spec-row"><div class="spec"><span class="spec-label">Full IMN model variables</span><span class="spec-value">13</span><span class="spec-caption">7 selected direct predictors + 6 composite indices</span></div>
    <div class="spec"><span class="spec-label">Complementary base learners</span><span class="spec-value">2</span><span class="spec-caption">Rotation Forest + Extra Trees</span></div>
    <div class="spec"><span class="spec-label">Primary decision threshold</span><span class="spec-value">0.6194</span><span class="spec-caption">Saved full IMN threshold · not 0.5</span></div></div></section>
    ''', unsafe_allow_html=True)


def render_inputs():
    with st.container(key="assessment"):
        st.markdown('<span id="assessment"></span>', unsafe_allow_html=True)
        intro("Assessment", "Enter the measurements.", "Use the stated units. Leave unavailable measurements blank; the saved imputer will complete them.")
        with st.container(key="measurement_card"):
            c1, c2, _ = st.columns([1.3, 1, 2])
            c1.button("Load synthetic example", on_click=reset_inputs, args=(True,), key="example")
            c2.button("Clear inputs", on_click=reset_inputs, args=(False,), key="clear")
            if st.session_state.get("example_active"):
                st.caption("Synthetic example loaded for interface testing. Replace all example values before entering another record.")
            st.markdown('<p class="form-note"><b>*</b> Selected direct predictor &nbsp; <b>†</b> Composite-index component<br>Other measurements support the saved imputation pathway. PA uses <b>mg/dL</b>.</p>', unsafe_allow_html=True)
            with st.form("prediction_form", border=False):
                values = {}
                tabs = st.tabs(["I · Inflammation", "M · Immune", "N · Nutrition"])
                for tab, group in zip(tabs, GROUP_LABELS):
                    with tab:
                        columns = st.columns(2, gap="large")
                        fields = [f for f in INPUT_FIELDS if f["group"] == group]
                        for i, field in enumerate(fields):
                            name = field["name"]
                            marker = "*" if name in SELECTED_DIRECT else ("†" if name in COMPOSITE_INPUTS else "")
                            label = f"{name}{marker} — {field['label']} ({field['unit']})"
                            if name == "CRP":
                                values[name] = columns[i % 2].text_input(label, key=f"input_{name}", placeholder="e.g. 8.7 or <5", help="CRP <x is converted to x/√2; >x uses x. Blank means missing.")
                            else:
                                values[name] = columns[i % 2].number_input(
                                    label, min_value=0.0, max_value=100.0 if field["unit"] == "%" else None,
                                    value=None, step=0.01, format="%.4f", key=f"input_{name}", placeholder="Not measured",
                                    help=f"Enter the measured value in {field['unit']}. Blank means missing, not zero.")
                submitted = st.form_submit_button("Generate prediction", type="primary")
            st.markdown('<p class="safety">Conditional classification between adenocarcinoma and squamous cell carcinoma.<br>Not a lung-cancer screening test or a substitute for pathology.</p>', unsafe_allow_html=True)
        if submitted:
            st.session_state.pop("single_result", None)
            st.session_state.pop("single_snapshot", None)
            try:
                with st.spinner("Applying the saved model…"):
                    st.session_state["single_result"] = score_frame(pd.DataFrame([values]))
                st.session_state["single_snapshot"] = dict(values)
            except InputError as exc:
                st.error(str(exc))
            except Exception:
                st.error("The model could not run. Verify the bundled model files and pinned dependencies. No prediction was produced.")
        return values


def render_results(values):
    payload = st.session_state.get("single_result")
    if payload is None:
        return
    result = payload["results"].iloc[0]
    p = float(result[f"{MAIN_LAYER}__adenocarcinoma_probability"])
    threshold = THRESHOLDS[MAIN_LAYER]
    predicted = result[f"{MAIN_LAYER}__class"]
    missing = int(result["missing_input_count"])
    st.markdown(f'''
    <section class="result-tile" id="result"><div class="tile-intro"><div class="tile-eyebrow">Your model output · I + M + N</div><h2>One result. More context.</h2></div>
    <div class="result-grid"><div class="prob-display"><div class="prob-label">Model-estimated adenocarcinoma probability</div>
    <div class="prob-number">{100*p:.1f}<span>%</span></div><div class="prob-rail"><div class="prob-fill" style="width:{100*p:.5f}%"></div>
    <div class="threshold" style="left:{100*threshold:.5f}%"></div><div class="threshold-label" style="left:{100*threshold:.5f}%">Threshold {threshold:.4f}</div></div></div>
    <div class="outcome-display"><div class="outcome-label">Predicted class at the saved threshold</div><div class="outcome-main">{html.escape(predicted)}</div>
    <div class="outcome-copy">Adenocarcinoma = 1 · Squamous cell carcinoma = 0<br>{27-missing} measured inputs · {missing} completed by the saved imputer<br>The probability is not a diagnostic certainty.</div></div></div></section>
    ''', unsafe_allow_html=True)
    with st.container(key="details"):
        st.caption("Results reflect the last submitted measurements. After changing values, select Generate prediction again.")
        cards = ""
        for layer in LAYERS:
            primary = " primary" if layer == MAIN_LAYER else ""
            layer_p = result[f"{layer}__adenocarcinoma_probability"]
            cards += f'<div class="layer-card{primary}"><div class="kicker">{LAYER_LABELS[layer]}{" · Primary" if primary else ""}</div><strong>{layer_p:.1%}</strong><div class="small">{result[f"{layer}__class"]}<br>Threshold {THRESHOLDS[layer]:.4f}</div></div>'
        st.markdown(f'<div class="layer-cards">{cards}</div>', unsafe_allow_html=True)
        st.markdown("##### Derived indices")
        formulas = ("NEUT / LYMPH", "NEUT × MONO / LYMPH", "LYMPH / MONO", "ALB + 5 × LYMPH", "ALB / (TP − ALB)", "GLU / ALB")
        index_html = "".join(f'<div class="index"><span class="index-name">{name}</span><span class="index-value">{result[name]:.3g}</span><span class="index-formula">{formula}</span></div>' for name, formula in zip(COMPOSITES, formulas))
        st.markdown(f'<div class="index-band">{index_html}</div>', unsafe_allow_html=True)
        if missing:
            st.info(f"{missing} of 27 inputs were missing and completed by the saved imputer: {result['imputed_fields'].replace(';', ', ')}. Interpret outputs cautiously when measurements are sparse.")
        if payload["notices"]["out_of_training_range_count"]:
            st.warning("Some inputs fall outside the saved development ranges. Values were not silently clipped or changed; check the units and measurements.")
        with st.expander("Review the exact inputs used"):
            review = pd.DataFrame({"Variable": DIRECT_COLUMNS, "Unit": [f["unit"] for f in INPUT_FIELDS],
                "Submitted": payload["input"].iloc[0].to_numpy(), "Used by model": payload["completed"].iloc[0].to_numpy(),
                "Imputed": payload["input"].iloc[0].isna().to_numpy()})
            st.dataframe(review, hide_index=True, width="stretch")
        st.download_button("Download this result", export_csv(payload["results"]), "imn_insight_prediction.csv", "text/csv", key="download_single", on_click="ignore")


def render_model():
    with st.container(key="model"):
        st.markdown('<span id="model"></span>', unsafe_allow_html=True)
        intro("Inside the model", "A consistent path from input to output.", "The interface applies the existing saved model. It does not retrain, tune, or select a new threshold.")
        st.markdown('''<div class="model-flow">
        <div class="flow-card"><div class="flow-step">01 / COMPLETE</div><div class="flow-title">Laboratory inputs.</div><div class="flow-copy">27 direct laboratory variables.<br>Missing values use the saved deterministic MICE-style imputer. Six prespecified indices are then calculated.</div></div>
        <div class="flow-card"><div class="flow-step">02 / COMBINE</div><div class="flow-title">Two learners.<br>One Stacking model.</div><div class="flow-copy">Rotation Forest and Extra Trees feed the saved L2 logistic meta-learner. This is not a simple average of probabilities.</div></div>
        <div class="flow-card"><div class="flow-step">03 / COMPARE</div><div class="flow-title">Three IMN layers.</div><div class="flow-copy">I: 4 variables<br>I + M: 6 variables<br>I + M + N: 13 variables<br>Each layer uses its own saved models and fixed decision threshold.</div></div></div>''', unsafe_allow_html=True)
        with st.expander("Input dictionary and model details"):
            dictionary = pd.DataFrame(INPUT_FIELDS).drop(columns="example").rename(columns={"name": "Variable", "label": "Measurement", "unit": "Unit", "group": "Domain"})
            st.dataframe(dictionary, hide_index=True, width="stretch")
            st.caption("TC accepts the CSV alias CHOL without numerical conversion. RDW means RDW-CV; PA is mg/dL. No age, sex, smoking history, or tumor markers enter the model.")
            st.caption("AGR uses ALB / (TP − ALB), not the independently measured GLB. Blank values are completed; no additional clipping or recalibration is applied by the interface.")
            st.code(f"Model SHA-256\n{MODEL_SHA256}", language=None)


def render_batch():
    with st.container(key="batch"):
        st.markdown('<span id="batch"></span>', unsafe_allow_html=True)
        intro("Batch assessment", "One file. The same model.", "Upload a de-identified CSV to generate all three layer outputs. Use the same units as the single-record form.")
        c1, c2 = st.columns(2)
        c1.download_button("Download blank template", input_template(), "imn_insight_input_template.csv", "text/csv", key="template", on_click="ignore")
        c2.download_button("Download synthetic example", export_csv(example_frame()), "imn_insight_synthetic_example.csv", "text/csv", key="example_csv", on_click="ignore")
        upload = st.file_uploader("CSV file · UTF-8 · up to 2,000 rows / 5 MB", type=["csv"], key="batch_upload", help="Do not upload names, identity numbers, or identifiable clinical records to a public service.")
        raw = upload.getvalue() if upload is not None else None
        fingerprint = hashlib.sha256(raw).hexdigest() if raw is not None else None
        if st.session_state.get("batch_fingerprint") != fingerprint:
            st.session_state.pop("batch_result", None)
        if st.button("Run batch prediction", disabled=upload is None, type="primary", key="run_batch"):
            st.session_state.pop("batch_result", None)
            try:
                with st.spinner("Validating inputs and applying the saved model…"):
                    st.session_state["batch_result"] = score_frame(read_csv_bytes(raw))
                    st.session_state["batch_fingerprint"] = fingerprint
            except InputError as exc:
                st.error(str(exc))
            except Exception:
                st.error("Batch prediction failed. Check the installed model dependencies. No partial result was exported.")
        payload = st.session_state.get("batch_result")
        if payload is not None:
            notices = payload["notices"]
            st.success(f"Completed {len(payload['results']):,} records in the original row order.")
            if notices["absent_columns"]:
                st.info("Absent columns were treated as missing: " + ", ".join(notices["absent_columns"]))
            if notices["ignored_columns"]:
                st.caption("Non-model columns were ignored and are not included in the output.")
            if notices["crp_converted"]:
                st.caption(f"Applied the prespecified CRP boundary rule to {notices['crp_converted']} value(s).")
            count = int(payload["results"]["missing_input_count"].gt(0).sum())
            if count:
                st.warning(f"{count} record(s) required imputation. The export includes a missing-value count and the imputed field names for every row.")
            if notices["out_of_training_range_count"]:
                st.warning("Some submitted values fall outside the saved development ranges. Check the measurements and units; no additional clipping was applied.")
            st.dataframe(safe_result_table(payload["results"]), hide_index=True, width="stretch")
            st.download_button("Download batch results", export_csv(payload["results"]), "imn_insight_batch_predictions.csv", "text/csv", key="batch_download", on_click="ignore")
        st.caption("Values are processed on the application server and held in the current session; the application does not write uploaded files or predictions to disk. Use only synthetic or appropriately de-identified data.")


render_hero()
values = render_inputs()
render_results(values)
render_model()
render_batch()
st.markdown(f'<footer class="footer"><b>IMN Insight · Adenocarcinoma vs Squamous</b><br>Educational model demonstration. Not for clinical diagnosis or treatment decisions.<br>Top-2 Stacking · Model fingerprint {MODEL_SHA256[:12]} · Interface 1.1</footer>', unsafe_allow_html=True)
