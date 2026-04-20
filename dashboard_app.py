from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import COUNTRIES

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def _read_csv(filename: str, required: bool = True) -> pd.DataFrame:
    path = OUTPUTS / filename
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required output file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    return {
        "rq1_mechanism": _read_csv("rq1_mechanism_breakdown.csv"),
        "rq1_platform": _read_csv("rq1_platform_profile.csv", required=False),
        "rq1_regime": _read_csv("rq1_regime_summary.csv", required=False),
        "rq2_daily": _read_csv("rq2_daily_series.csv"),
        "rq2_escalation": _read_csv("rq2_escalation_patterns.csv"),
        "rq2_recovery": _read_csv("rq2_recovery_timing.csv", required=False),
        "phase2_event_metrics": _read_csv("phase2_event_model_metrics.csv", required=False),
        "phase2_nextday_metrics": _read_csv("phase2_nextday_model_metrics.csv", required=False),
        "phase2_event_preds": _read_csv("phase2_event_risk_predictions.csv", required=False),
        "phase2_nextday_preds": _read_csv("phase2_nextday_predictions.csv", required=False),
    }


def _quality_confidence(event_daily: pd.DataFrame) -> tuple[str, list[str]]:
    warnings: list[str] = []
    n_days = int(event_daily["t_day"].nunique())
    med_n = float(event_daily["measurement_count"].median())
    low_days = int((event_daily["measurement_count"] < 10).sum())
    td = set(event_daily["t_day"].astype(int).tolist())
    miss_t1 = 1 not in td
    miss_t2 = 2 not in td

    score = 0
    if n_days >= 18:
        score += 1
    else:
        warnings.append("Limited day coverage around event.")

    if med_n >= 20:
        score += 1
    else:
        warnings.append("Low median daily sample size.")

    if low_days <= 2:
        score += 1
    else:
        warnings.append("Many low-sample days (n < 10).")

    if not miss_t1:
        score += 1
    else:
        warnings.append("Missing T+1 measurements.")

    if not miss_t2:
        score += 1
    else:
        warnings.append("Missing T+2 measurements.")

    if score >= 4:
        return "HIGH", warnings
    if score >= 3:
        return "MEDIUM", warnings
    return "LOW", warnings


def estimate_recovery_eta(
    event_daily: pd.DataFrame,
    std_multiplier: float,
    min_n: int,
    consecutive_days: int,
) -> dict[str, str]:
    d = event_daily.sort_values("t_day").copy()
    base = d[d["t_day"].between(-14, -8)]
    if len(base) < 3:
        return {
            "eta_range": "N/A",
            "method": "insufficient baseline",
            "warning": "Not enough baseline days to estimate threshold.",
        }

    base_mean = float(base["hard_block_frac"].mean())
    base_std = float(base["hard_block_frac"].std(ddof=0)) if len(base) > 1 else 0.0
    threshold = base_mean + std_multiplier * base_std

    post = d[d["t_day"] >= 0].copy()
    post = post[post["measurement_count"] >= min_n].copy()

    if post.empty:
        return {
            "eta_range": "N/A",
            "method": "insufficient post-event volume",
            "warning": "No reliable post-event days at chosen min sample threshold.",
        }

    ts = post["t_day"].astype(int).tolist()
    vals = post.set_index(post["t_day"].astype(int))["hard_block_frac"].to_dict()

    for i in range(len(ts) - consecutive_days + 1):
        window = ts[i : i + consecutive_days]
        if window[-1] - window[0] != consecutive_days - 1:
            continue
        if all(vals[t] <= threshold for t in window):
            return {
                "eta_range": f"T+{window[0]} to T+{window[-1]}",
                "method": "observed threshold crossing",
                "warning": "",
            }

    fit = post[post["t_day"] >= 3][["t_day", "hard_block_frac", "measurement_count"]].dropna()
    if len(fit) < 4:
        return {
            "eta_range": "N/A",
            "method": "insufficient trend points",
            "warning": "Too few reliable post-event days for forecast fallback.",
        }

    x = fit["t_day"].to_numpy(dtype=float)
    y = fit["hard_block_frac"].to_numpy(dtype=float)
    w = np.sqrt(fit["measurement_count"].to_numpy(dtype=float))

    try:
        slope, intercept = np.polyfit(x, y, 1, w=w)
    except Exception:
        return {
            "eta_range": "N/A",
            "method": "trend fit failure",
            "warning": "Forecast fit failed for this scenario.",
        }

    if slope >= 0:
        return {
            "eta_range": "N/A",
            "method": "non-declining trend",
            "warning": "Estimated hard-block trend is not declining.",
        }

    t_hit = (threshold - intercept) / slope
    if np.isnan(t_hit):
        return {
            "eta_range": "N/A",
            "method": "invalid crossing",
            "warning": "Threshold crossing could not be computed.",
        }

    lo = int(max(np.floor(t_hit), 0))
    hi = int(max(np.ceil(t_hit), lo))
    return {
        "eta_range": f"T+{lo} to T+{hi}",
        "method": "weighted linear fallback forecast",
        "warning": "Forecast-based estimate only; treat as indicative.",
    }


def simulate_event_scenario(
    event_daily: pd.DataFrame,
    historical_pre_hard_frac: float,
    historical_volume_ratio: float,
    projected_hard_frac: float,
    projected_volume_ratio: float,
) -> pd.DataFrame:
    sim = event_daily.copy()

    hard_delta = projected_hard_frac - historical_pre_hard_frac
    sim["hard_block_frac"] = (sim["hard_block_frac"] + hard_delta).clip(0.0, 1.0)

    if historical_volume_ratio > 0:
        volume_factor = projected_volume_ratio / historical_volume_ratio
    else:
        volume_factor = 1.0

    sim["measurement_count"] = np.maximum(
        np.round(sim["measurement_count"] * volume_factor).astype(int),
        1,
    )
    return sim


def compute_risk_level(
    projected_hard_frac: float,
    projected_volume_ratio: float,
    baseline_hard_frac: float,
    historical_pre_hard_frac: float,
) -> tuple[str, int, str]:
    score = 0

    if projected_volume_ratio < 0.8:
        score += 2
    if projected_volume_ratio < 0.6:
        score += 1

    if projected_hard_frac >= baseline_hard_frac * 1.2:
        score += 2
    if projected_hard_frac >= max(historical_pre_hard_frac, 1e-6) * 1.15:
        score += 1

    if score >= 5:
        return "HIGH", score, "Escalation likely"
    if score >= 3:
        return "MEDIUM", score, "Escalation possible"
    return "LOW", score, "Escalation less likely"


def main() -> None:
    st.set_page_config(
        page_title="OONI EAC Interference Dashboard",
        page_icon="📶",
        layout="wide",
    )

    st.title("OONI EAC Social Media Interference Dashboard")
    st.caption(
        "Interactive dashboard for RQ1 mechanism fingerprints and RQ2 escalation/recovery signals "
        "around electoral events in Kenya, Uganda, and Tanzania."
    )

    try:
        data = load_data()
    except Exception as exc:
        st.error(f"Could not load output files: {exc}")
        st.stop()

    rq1_mech = data["rq1_mechanism"]
    rq1_platform = data["rq1_platform"]
    rq2_daily = data["rq2_daily"]
    rq2_esc = data["rq2_escalation"]
    rq2_rec = data["rq2_recovery"]
    phase2_event_metrics = data["phase2_event_metrics"]
    phase2_nextday_metrics = data["phase2_nextday_metrics"]
    phase2_event_preds = data["phase2_event_preds"]
    phase2_nextday_preds = data["phase2_nextday_preds"]

    st.sidebar.header("Dashboard Filters")
    st.sidebar.caption(
        "Usable RQ2 data means we found enough OONI checks around an event date "
        "(before and after) to calculate escalation trends."
    )

    configured_country_codes = list(COUNTRIES.keys())
    available_country_codes = set(rq2_esc["country"].dropna().unique().tolist())
    country_label_map = {
        code: (
            f"{COUNTRIES.get(code, code)} [usable RQ2 data]"
            if code in available_country_codes
            else f"{COUNTRIES.get(code, code)} [no usable RQ2 data]"
        )
        for code in configured_country_codes
    }
    default_idx = next(
        (i for i, cc in enumerate(configured_country_codes) if cc in available_country_codes),
        0,
    )
    selected_country_code = st.sidebar.selectbox(
        "Country",
        configured_country_codes,
        index=default_idx,
        format_func=lambda cc: country_label_map.get(cc, cc),
    )
    selected_country = COUNTRIES.get(selected_country_code, selected_country_code)

    coverage_rows = []
    for cc in configured_country_codes:
        c_name = COUNTRIES.get(cc, cc)
        c_daily = rq2_daily[rq2_daily["event_country"] == cc]
        c_esc = rq2_esc[rq2_esc["country"] == cc]
        has_usable = not c_esc.empty
        coverage_rows.append(
            {
                "country": c_name,
                "rq2_status": "usable" if has_usable else "no usable data",
                "events_with_escalation_rows": int(c_esc["event_label"].nunique()) if not c_esc.empty else 0,
                "event_day_rows_found": int(len(c_daily)),
            }
        )

    st.markdown("### Data Coverage")
    st.caption(
        "A country is marked usable when at least one event has enough event-window "
        "measurements to produce RQ2 escalation rows."
    )
    st.dataframe(pd.DataFrame(coverage_rows), hide_index=True, use_container_width=True)

    esc_country = rq2_esc[rq2_esc["country"] == selected_country_code].copy()
    events = sorted(esc_country["event_label"].dropna().unique().tolist())
    if not events:
        st.warning(
            f"{selected_country} currently has no usable RQ2 event windows in this dataset. "
            "Try another country with '[usable RQ2 data]' in the sidebar."
        )
        st.stop()

    selected_event = st.sidebar.selectbox("Electoral Event", events)

    day_min = int(rq2_daily["t_day"].min())
    day_max = int(rq2_daily["t_day"].max())
    selected_window = st.sidebar.slider(
        "Visible day window",
        min_value=day_min,
        max_value=day_max,
        value=(max(-14, day_min), min(7, day_max)),
        step=1,
    )

    mechanisms = sorted(rq1_mech["blocking_mechanism"].dropna().unique().tolist())
    selected_mechanisms = st.sidebar.multiselect(
        "RQ1 mechanism filter",
        options=mechanisms,
        default=mechanisms,
    )

    event_row = esc_country[esc_country["event_label"] == selected_event].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Volume Ratio (Pre/Baseline)", f"{float(event_row['volume_ratio']):.2f}")
    m2.metric("Hard-Block Ratio", f"{float(event_row['hard_block_ratio']):.2f}")
    m3.metric("Escalation Flag", "YES" if bool(event_row["escalation_flag"]) else "NO")
    m4.metric("Regime Type", str(event_row["regime_type"]).replace("_", " ").title())

    st.markdown("### Phase 2 Retrained Models")
    if (
        phase2_event_metrics.empty
        or phase2_nextday_metrics.empty
        or phase2_event_preds.empty
        or phase2_nextday_preds.empty
    ):
        st.info(
            "Phase 2 model outputs were not found. Run the pipeline to generate them "
            "(example: python main.py --skip-collection)."
        )
    else:
        pm1, pm2 = st.columns([1.0, 1.1])

        with pm1:
            st.caption("Backtest metrics (leave-one-event-out)")
            metric_cols = [
                "model_name",
                "f1",
                "precision",
                "recall",
                "roc_auc",
                "avg_precision",
                "is_selected",
            ]
            metric_cols = [c for c in metric_cols if c in phase2_event_metrics.columns]
            st.dataframe(
                phase2_event_metrics[metric_cols].sort_values("f1", ascending=False),
                hide_index=True,
                use_container_width=True,
            )

            nd_metric_cols = [c for c in metric_cols if c in phase2_nextday_metrics.columns]
            st.dataframe(
                phase2_nextday_metrics[nd_metric_cols].sort_values("f1", ascending=False),
                hide_index=True,
                use_container_width=True,
            )

        with pm2:
            event_prob = phase2_event_preds[
                (phase2_event_preds["event_label"] == selected_event)
                & (phase2_event_preds["event_country"] == event_row["country"])
            ]
            if not event_prob.empty:
                p = float(event_prob.iloc[0]["escalation_prob"])
                st.metric("Model Escalation Probability", f"{p:.2%}")
                st.caption(
                    f"Event model prediction: {'ESCALATION' if p >= 0.5 else 'NO ESCALATION'}"
                )

            nd_probs = phase2_nextday_preds[
                (phase2_nextday_preds["event_label"] == selected_event)
                & (phase2_nextday_preds["event_country"] == event_row["country"])
            ].copy()

            if nd_probs.empty:
                st.info("No next-day model rows found for this event.")
            else:
                nd_probs = nd_probs.sort_values("t_day")
                latest = nd_probs.iloc[-1]
                st.metric("Latest Next-Day Warning Probability", f"{float(latest['nextday_warning_prob']):.2%}")

                fig_prob = px.line(
                    nd_probs,
                    x="t_day",
                    y="nextday_warning_prob",
                    markers=True,
                    labels={
                        "t_day": "Days relative to event",
                        "nextday_warning_prob": "P(next-day hard-block jump)",
                    },
                    title="Next-day warning probability by day",
                )
                fig_prob.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=320)
                fig_prob.update_yaxes(range=[0, 1])
                st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("### RQ2 Event Timeline")
    event_daily = rq2_daily[
        (rq2_daily["event_label"] == selected_event)
        & (rq2_daily["event_country"] == event_row["country"])
        & (rq2_daily["t_day"].between(selected_window[0], selected_window[1]))
    ].copy()

    if event_daily.empty:
        st.warning("No daily rows available for this event and window.")
    else:
        event_daily["hard_pct"] = event_daily["hard_block_frac"] * 100.0
        fig_timeline = go.Figure()
        fig_timeline.add_trace(
            go.Bar(
                x=event_daily["t_day"],
                y=event_daily["measurement_count"],
                name="Measurements",
                marker_color="#4C78A8",
                opacity=0.25,
                yaxis="y2",
            )
        )
        fig_timeline.add_trace(
            go.Scatter(
                x=event_daily["t_day"],
                y=event_daily["hard_pct"],
                name="Hard-block fraction (%)",
                mode="lines+markers",
                line=dict(color="#C00000", width=2),
            )
        )

        fig_timeline.add_vline(x=0, line_dash="dash", line_color="black")
        fig_timeline.update_layout(
            xaxis_title="Days relative to event",
            yaxis_title="Hard-block fraction (%)",
            yaxis=dict(range=[0, 100]),
            yaxis2=dict(
                title="Daily measurements",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=40, b=10),
            height=420,
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    st.markdown("### RQ1 Mechanism Fingerprints")
    rq1_view = rq1_mech[
        (rq1_mech["country_name"] == selected_country)
        & (rq1_mech["blocking_mechanism"].isin(selected_mechanisms))
    ].copy()
    rq1_view["proportion_pct"] = rq1_view["proportion"] * 100.0

    if rq1_view.empty:
        st.warning("No mechanism rows available for this filter combination.")
    else:
        fig_mech = px.bar(
            rq1_view.sort_values("proportion_pct", ascending=False),
            x="blocking_mechanism",
            y="proportion_pct",
            color="blocking_mechanism",
            title=f"Mechanism composition for {selected_country}",
            labels={"proportion_pct": "Share of anomalies (%)", "blocking_mechanism": "Mechanism"},
        )
        fig_mech.update_layout(showlegend=False, margin=dict(l=10, r=10, t=45, b=10), height=380)
        st.plotly_chart(fig_mech, use_container_width=True)

    col_a, col_b = st.columns([1.15, 1.0])

    with col_a:
        st.markdown("### Event Comparison Table")
        table_cols = [
            "event_label",
            "volume_ratio",
            "hard_block_ratio",
            "volume_drop_flag",
            "mechanism_shift_flag",
            "escalation_flag",
        ]
        st.dataframe(
            esc_country[table_cols].sort_values("event_label"),
            use_container_width=True,
            hide_index=True,
        )

    with col_b:
        st.markdown("### Recovery Timing")
        if not rq2_rec.empty:
            rec_view = rq2_rec[rq2_rec["country"] == event_row["country"]].copy()
            if rec_view.empty:
                st.info("Recovery file found but no rows for selected country.")
            else:
                if "recovery_observed" not in rec_view.columns:
                    rec_view["recovery_observed"] = rec_view.get("recovered_within_window", False)
                if "is_censored" not in rec_view.columns:
                    rec_view["is_censored"] = ~rec_view["recovery_observed"].astype(bool)

                observed_count = int(rec_view["recovery_observed"].astype(bool).sum())
                censored_count = int(rec_view["is_censored"].astype(bool).sum())
                c1, c2 = st.columns(2)
                c1.metric("Observed recoveries", observed_count)
                c2.metric("Censored events", censored_count)

                if {
                    "volume_recovery_threshold",
                    "baseline_volume",
                    "hard_block_recovery_threshold",
                    "min_consecutive_days",
                }.issubset(rec_view.columns):
                    sample = rec_view.dropna(subset=["baseline_volume", "volume_recovery_threshold"]).head(1)
                    if not sample.empty and float(sample.iloc[0]["baseline_volume"]) > 0:
                        ratio = float(sample.iloc[0]["volume_recovery_threshold"]) / float(sample.iloc[0]["baseline_volume"])
                        hard_thr = float(sample.iloc[0]["hard_block_recovery_threshold"])
                        min_days = int(sample.iloc[0]["min_consecutive_days"])
                        st.caption(
                            f"Criteria in use: volume >= baseline x {ratio:.2f}, "
                            f"hard-block <= {hard_thr:.2f}, consecutive days = {min_days}."
                        )

                display_cols = [
                    "event_label",
                    "recovery_observed",
                    "is_censored",
                    "days_to_recovery",
                    "max_observed_t_day",
                    "baseline_volume",
                    "volume_recovery_threshold",
                    "hard_block_recovery_threshold",
                    "min_consecutive_days",
                ]
                display_cols = [c for c in display_cols if c in rec_view.columns]
                st.dataframe(rec_view[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info(
                "No precomputed recovery file found, so recovery ETA in the simulator below "
                "will be estimated from daily time series."
            )

    st.markdown("---")
    st.markdown("## Scenario Simulator")
    st.caption(
        "Use constrained input fields to test what-if conditions while keeping values in valid ranges."
    )

    s1, s2, s3, s4 = st.columns(4)
    projected_hard = s1.slider(
        "Projected pre-event hard-block fraction",
        min_value=0.0,
        max_value=1.0,
        value=float(np.clip(event_row["pre_event_hard_frac"], 0.0, 1.0)),
        step=0.01,
        format="%.2f",
    )
    projected_volume = s2.slider(
        "Projected pre-event volume ratio",
        min_value=0.20,
        max_value=2.00,
        value=float(np.clip(event_row["volume_ratio"], 0.20, 2.00)),
        step=0.01,
        format="%.2f",
    )
    recovery_mode = s3.selectbox(
        "Recovery strictness",
        options=["Conservative", "Balanced", "Optimistic"],
        index=1,
    )
    min_daily_n = s4.number_input(
        "Minimum reliable daily sample",
        min_value=5,
        max_value=100,
        value=10,
        step=1,
    )

    require_two_days = st.checkbox("Require 2 consecutive recovery days", value=True)

    mode_to_std = {
        "Conservative": 0.50,
        "Balanced": 0.75,
        "Optimistic": 1.00,
    }

    if st.button("Evaluate Scenario", type="primary"):
        event_full = rq2_daily[
            (rq2_daily["event_label"] == selected_event)
            & (rq2_daily["event_country"] == event_row["country"])
        ][["t_day", "measurement_count", "hard_block_frac"]].copy()

        if event_full.empty:
            st.error("Cannot simulate because no daily data exists for this event.")
        else:
            sim = simulate_event_scenario(
                event_daily=event_full,
                historical_pre_hard_frac=float(event_row["pre_event_hard_frac"]),
                historical_volume_ratio=float(event_row["volume_ratio"]),
                projected_hard_frac=float(projected_hard),
                projected_volume_ratio=float(projected_volume),
            )

            risk_level, risk_score, escalation_outlook = compute_risk_level(
                projected_hard_frac=float(projected_hard),
                projected_volume_ratio=float(projected_volume),
                baseline_hard_frac=float(event_row["baseline_hard_frac"]),
                historical_pre_hard_frac=float(event_row["pre_event_hard_frac"]),
            )

            eta = estimate_recovery_eta(
                event_daily=sim,
                std_multiplier=mode_to_std[recovery_mode],
                min_n=int(min_daily_n),
                consecutive_days=2 if require_two_days else 1,
            )

            confidence, conf_warnings = _quality_confidence(sim)
            warning_text = " | ".join(conf_warnings + ([eta["warning"]] if eta["warning"] else []))

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Risk Level", risk_level)
            r2.metric("Risk Score", str(risk_score))
            r3.metric("Escalation Outlook", escalation_outlook)
            r4.metric("Recovery ETA", eta["eta_range"])

            st.write(f"Estimator method: {eta['method']}")
            st.write(f"Confidence: {confidence}")
            if warning_text:
                st.warning(warning_text)

            sim["hard_pct"] = sim["hard_block_frac"] * 100.0
            fig_sim = px.line(
                sim.sort_values("t_day"),
                x="t_day",
                y="hard_pct",
                markers=True,
                labels={"t_day": "Days relative to event", "hard_pct": "Hard-block fraction (%)"},
                title="Simulated hard-block trajectory",
            )
            fig_sim.add_vline(x=0, line_dash="dash", line_color="black")
            fig_sim.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=340)
            st.plotly_chart(fig_sim, use_container_width=True)

    st.markdown("---")
    st.caption(
        "Decision-support only. Use alongside contextual evidence and domain expert judgment."
    )


if __name__ == "__main__":
    main()
