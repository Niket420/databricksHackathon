"""
UPI Fraud Detection — Streamlit App
Supports:
  • Step-by-step manual transaction entry
  • Bulk CSV upload with simulated streaming
  • Graph feature extraction (NetworkX, pure Python — no Spark dependency)
  • Databricks Model Serving inference
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import requests
import json
import time
import io
from datetime import datetime, timedelta
import random

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UPI Fraud Radar",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Dark cyber background */
.stApp {
    background: #080c14;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1420 !important;
    border-right: 1px solid #1e3a5f;
}

/* Header strip */
.hero-header {
    background: linear-gradient(135deg, #0a1628 0%, #0d2137 50%, #071020 100%);
    border: 1px solid #1e4d7b;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(0,200,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00d4ff;
    letter-spacing: -1px;
    margin: 0;
}
.hero-sub {
    color: #64748b;
    font-size: 0.9rem;
    margin-top: 6px;
    font-weight: 300;
}

/* Metric cards */
.metric-card {
    background: #0d1824;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #00d4ff55; }
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00d4ff;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Feature table */
.feature-table {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
}

/* Result cards */
.fraud-card {
    background: linear-gradient(135deg, #2d0a0a, #1a0505);
    border: 1px solid #ff4444;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 6px 0;
}
.legit-card {
    background: linear-gradient(135deg, #0a2d1a, #051a0d);
    border: 1px solid #00ff88;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 6px 0;
}
.warn-card {
    background: linear-gradient(135deg, #2d1f0a, #1a1005);
    border: 1px solid #ffaa00;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 6px 0;
}

/* Log terminal */
.terminal {
    background: #050810;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 16px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #00ff88;
    height: 220px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
}

/* Input styling */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: #0d1824 !important;
    border: 1px solid #1e3a5f !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #0077cc) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    letter-spacing: 0.5px;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Progress bar colour override */
.stProgress > div > div > div { background: #00d4ff !important; }

/* Tab styling */
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #64748b;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom-color: #00d4ff !important;
}

/* Dataframe override */
.dataframe { font-family: 'Space Mono', monospace; font-size: 0.72rem; }

/* Section headers */
.section-hdr {
    font-family: 'Space Mono', monospace;
    color: #00d4ff;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 8px;
    margin: 20px 0 14px 0;
}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ───────────────────────────────────────────────────────────────
for key, default in {
    "transactions": [],
    "graph_features": pd.DataFrame(),
    "predictions": [],
    "log_lines": [],
    "stream_running": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── HELPERS ────────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.log_lines.append(f"[{ts}] {msg}")
    if len(st.session_state.log_lines) > 120:
        st.session_state.log_lines = st.session_state.log_lines[-120:]


def extract_graph_features(tx_list: list[dict]) -> pd.DataFrame:
    """Build a directed graph and extract per-node ML features."""
    if not tx_list:
        return pd.DataFrame()

    G = nx.DiGraph()
    for tx in tx_list:
        src, dst, amt = tx["sender"], tx["receiver"], float(tx["amount"])
        if G.has_edge(src, dst):
            G[src][dst]["weight"] += amt
            G[src][dst]["count"] += 1
        else:
            G.add_edge(src, dst, weight=amt, count=1)

    rows = []
    for node in G.nodes():
        in_edges  = list(G.in_edges(node, data=True))
        out_edges = list(G.out_edges(node, data=True))

        tx_count_in  = sum(d["count"]  for _, _, d in in_edges)
        tx_count_out = sum(d["count"]  for _, _, d in out_edges)
        amount_in    = sum(d["weight"] for _, _, d in in_edges)
        amount_out   = sum(d["weight"] for _, _, d in out_edges)
        degree       = G.degree(node)

        # Triangle count (undirected proxy)
        try:
            tri = nx.triangles(G.to_undirected(), node)
        except Exception:
            tri = 0

        # Cluster size  = weakly connected component size
        wcc = nx.node_connected_component(G.to_undirected(), node)
        cluster_size = len(wcc)

        rows.append({
            "id":             node,
            "degree":         degree,
            "tx_count_in":    tx_count_in,
            "tx_count_out":   tx_count_out,
            "amount_in":      round(amount_in,  4),
            "amount_out":     round(amount_out, 4),
            "cluster_size":   cluster_size,
            "triangle_count": tri,
        })

    return pd.DataFrame(rows)


def call_databricks_model(endpoint_url: str, token: str, features_df: pd.DataFrame) -> list:
    """
    POST to Databricks Model Serving endpoint.
    Expects endpoint to accept {'dataframe_records': [...]} format.
    """
    records = features_df.to_dict(orient="records")
    payload = {"dataframe_records": records}

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }
    try:
        resp = requests.post(endpoint_url, headers=headers,
                             data=json.dumps(payload), timeout=30)
        resp.raise_for_status()
        result = resp.json()
        # Databricks returns {"predictions": [...]}
        preds = result.get("predictions", result.get("outputs", []))
        log(f"✅ Model returned {len(preds)} prediction(s)")
        return preds
    except requests.exceptions.ConnectionError:
        log("❌ Connection error — is the endpoint URL correct?")
    except requests.exceptions.HTTPError as e:
        log(f"❌ HTTP {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        log(f"❌ Unexpected error: {e}")
    return []


def mock_predict(features_df: pd.DataFrame) -> list:
    """Demo predictions when no real endpoint is configured."""
    results = []
    for _, row in features_df.iterrows():
        # Heuristic: high out-degree + low in-amount + big out-amount → suspicious
        score = 0.0
        if row["tx_count_out"] > 5:          score += 0.25
        if row["amount_out"] > 500_000:       score += 0.25
        if row["amount_in"] < 1_000:          score += 0.2
        if row["triangle_count"] == 0:        score += 0.15
        if row["cluster_size"] == 1:          score += 0.15
        score = min(score + random.uniform(-0.05, 0.05), 1.0)
        results.append(round(score, 4))
    return results


def fraud_label(score: float) -> tuple[str, str]:
    if score >= 0.7:
        return "🚨 FRAUD",   "fraud-card"
    elif score >= 0.4:
        return "⚠️ SUSPECT", "warn-card"
    else:
        return "✅ LEGIT",   "legit-card"


# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Config")
    endpoint_url = st.text_input(
        "Databricks Endpoint URL",
        placeholder="https://<workspace>.cloud.databricks.com/serving-endpoints/<name>/invocations",
        help="Your Databricks Model Serving invocation URL",
    )
    db_token = st.text_input(
        "Databricks Token",
        type="password",
        placeholder="dapiXXXXXXXXXXXXXX",
    )
    use_mock = st.checkbox("Use mock predictions (demo mode)", value=True,
                           help="Enable when endpoint is still deploying")

    st.divider()
    st.markdown("### 📊 Session Stats")
    n_tx   = len(st.session_state.transactions)
    n_accs = len(set(
        [t["sender"] for t in st.session_state.transactions] +
        [t["receiver"] for t in st.session_state.transactions]
    ))
    n_fraud = sum(1 for p in st.session_state.predictions if p >= 0.7)

    st.markdown(f"""
    <div class="metric-card" style="margin-bottom:10px">
        <div class="metric-value">{n_tx}</div>
        <div class="metric-label">Transactions</div>
    </div>
    <div class="metric-card" style="margin-bottom:10px">
        <div class="metric-value">{n_accs}</div>
        <div class="metric-label">Unique Accounts</div>
    </div>
    <div class="metric-card">
        <div class="metric-value" style="color:#ff4444">{n_fraud}</div>
        <div class="metric-label">Fraud Alerts</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear All Data"):
        for k in ["transactions", "graph_features", "predictions", "log_lines"]:
            st.session_state[k] = [] if k != "graph_features" else pd.DataFrame()
        st.rerun()

# ─── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <p class="hero-title">🔍 UPI FRAUD RADAR</p>
    <p class="hero-sub">Graph-feature extraction · Databricks model serving · Real-time streaming</p>
</div>
""", unsafe_allow_html=True)

# ─── TABS ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "➕  Manual Entry",
    "📂  Bulk / Stream CSV",
    "🕸️  Graph Features",
    "🤖  Predictions",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — MANUAL ENTRY
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-hdr">Add Transaction</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    sender    = c1.text_input("Sender ID",   placeholder="e.g. C1232585256")
    receiver  = c2.text_input("Receiver ID", placeholder="e.g. M98765432")
    amount    = c3.number_input("Amount (₹)", min_value=0.01, value=1000.0, step=0.01)

    c4, c5 = st.columns(2)
    ts_date   = c4.date_input("Date", value=datetime.today())
    ts_time   = c5.time_input("Time", value=datetime.now().time())
    timestamp = datetime.combine(ts_date, ts_time).isoformat()

    tx_type   = st.selectbox("Transaction Type",
                             ["P2P", "P2M", "Recharge", "Bill Payment", "Other"])

    btn_add, btn_run = st.columns([1, 1])
    if btn_add.button("➕ Add Transaction"):
        if not sender or not receiver:
            st.warning("Please fill in Sender and Receiver IDs.")
        else:
            tx = {"sender": sender, "receiver": receiver,
                  "amount": amount, "timestamp": timestamp,
                  "type": tx_type}
            st.session_state.transactions.append(tx)
            log(f"Added TX: {sender} → {receiver}  ₹{amount:,.2f}")
            st.success(f"Transaction added! Total: {len(st.session_state.transactions)}")

    if btn_run.button("🚀 Extract Features + Predict"):
        if not st.session_state.transactions:
            st.warning("Add at least one transaction first.")
        else:
            with st.spinner("Building graph & extracting features…"):
                feats = extract_graph_features(st.session_state.transactions)
                st.session_state.graph_features = feats
                log(f"Graph built: {len(feats)} nodes extracted")

            with st.spinner("Running model inference…"):
                if use_mock or not endpoint_url or not db_token:
                    preds = mock_predict(feats)
                    log("Using mock predictor (demo mode)")
                else:
                    preds = call_databricks_model(endpoint_url, db_token, feats)
                st.session_state.predictions = preds
                n_f = sum(1 for p in preds if p >= 0.7)
                log(f"Inference complete. Fraud alerts: {n_f}/{len(preds)}")
            st.success("Done! Check the Predictions tab.")
            st.rerun()

    # Recent transactions table
    if st.session_state.transactions:
        st.markdown('<p class="section-hdr">Recent Transactions</p>', unsafe_allow_html=True)
        df_show = pd.DataFrame(st.session_state.transactions[-20:][::-1])
        st.dataframe(df_show, use_container_width=True, height=260)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — BULK / STREAM CSV
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-hdr">Upload CSV</p>', unsafe_allow_html=True)
    st.markdown("""
    Expected columns: **`sender`, `receiver`, `amount`, `timestamp`**  
    Optional: `type`  
    """)

    # Sample CSV download
    sample = pd.DataFrame({
        "sender":    ["C1001", "C1002", "C1003", "C1004", "C1005"],
        "receiver":  ["M2001", "M2002", "C1001", "M2001", "C1003"],
        "amount":    [250.0, 12500.0, 800.0, 50000.0, 999.99],
        "timestamp": [(datetime.now() - timedelta(minutes=i*5)).isoformat() for i in range(5)],
        "type":      ["P2M", "P2M", "P2P", "P2M", "P2P"],
    })
    csv_bytes = sample.to_csv(index=False).encode()
    st.download_button("⬇️ Download sample CSV", csv_bytes,
                       "sample_upi.csv", "text/csv")

    uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])

    stream_speed = st.slider("Streaming speed (rows/sec)", 1, 50, 5)

    col_stream, col_bulk = st.columns(2)

    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.info(f"Loaded **{len(df_up):,}** rows · columns: {list(df_up.columns)}")
        required = {"sender", "receiver", "amount"}
        missing  = required - set(df_up.columns)
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            # ── STREAM ──────────────────────────────────────────────────────
            if col_stream.button("▶️ Stream transactions"):
                progress_bar = st.progress(0)
                status_text  = st.empty()
                result_area  = st.empty()
                total        = len(df_up)

                st.session_state.transactions = []  # fresh run
                batch_size   = max(1, stream_speed)
                all_preds    = []
                all_feats    = pd.DataFrame()

                for i in range(0, total, batch_size):
                    batch = df_up.iloc[i:i+batch_size]
                    for _, row in batch.iterrows():
                        tx = {
                            "sender":    str(row["sender"]),
                            "receiver":  str(row["receiver"]),
                            "amount":    float(row["amount"]),
                            "timestamp": str(row.get("timestamp", datetime.now().isoformat())),
                            "type":      str(row.get("type", "P2P")),
                        }
                        st.session_state.transactions.append(tx)

                    # extract features on rolling window
                    feats  = extract_graph_features(st.session_state.transactions)
                    preds  = mock_predict(feats) if (use_mock or not endpoint_url or not db_token) \
                             else call_databricks_model(endpoint_url, db_token, feats)

                    all_feats = feats
                    all_preds = preds
                    pct       = min((i + batch_size) / total, 1.0)
                    n_done    = min(i + batch_size, total)

                    progress_bar.progress(pct)
                    n_f = sum(1 for p in preds if p >= 0.7)
                    status_text.markdown(
                        f"**Processed:** {n_done:,}/{total:,} rows &nbsp;|&nbsp; "
                        f"**Fraud alerts so far:** 🚨 {n_f}"
                    )
                    result_area.dataframe(
                        feats.assign(fraud_score=preds)
                            .sort_values("fraud_score", ascending=False)
                            .head(8),
                        use_container_width=True,
                    )
                    log(f"Stream batch {i}–{n_done}: {n_f} fraud(s) detected")
                    time.sleep(1.0 / stream_speed)

                st.session_state.graph_features = all_feats
                st.session_state.predictions    = all_preds
                st.success("✅ Stream complete! Check Predictions tab.")

            # ── BULK ────────────────────────────────────────────────────────
            if col_bulk.button("⚡ Bulk Process All"):
                with st.spinner("Processing all rows…"):
                    txs = []
                    for _, row in df_up.iterrows():
                        txs.append({
                            "sender":    str(row["sender"]),
                            "receiver":  str(row["receiver"]),
                            "amount":    float(row["amount"]),
                            "timestamp": str(row.get("timestamp", "")),
                            "type":      str(row.get("type", "P2P")),
                        })
                    st.session_state.transactions = txs
                    feats = extract_graph_features(txs)
                    st.session_state.graph_features = feats
                    preds = mock_predict(feats) if (use_mock or not endpoint_url or not db_token) \
                            else call_databricks_model(endpoint_url, db_token, feats)
                    st.session_state.predictions = preds
                    log(f"Bulk: {len(txs)} TXs, {len(feats)} nodes, "
                        f"{sum(1 for p in preds if p >= 0.7)} fraud alerts")
                st.success("✅ Bulk processing done! Check Predictions tab.")
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — GRAPH FEATURES
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-hdr">Extracted Node Features</p>', unsafe_allow_html=True)
    feats_df = st.session_state.graph_features

    if feats_df.empty:
        st.info("No features yet. Add transactions and click **Extract Features + Predict**.")
    else:
        # Summary row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Nodes",        len(feats_df))
        m2.metric("Max Degree",   int(feats_df["degree"].max()))
        m3.metric("Max Amount Out", f"₹{feats_df['amount_out'].max():,.0f}")
        m4.metric("Isolated Nodes",
                  int((feats_df["triangle_count"] == 0).sum()))

        st.markdown('<p class="section-hdr">Feature Table</p>', unsafe_allow_html=True)
        st.dataframe(
            feats_df.style.background_gradient(
                subset=["degree", "tx_count_out", "amount_out"],
                cmap="YlOrRd"
            ).format({
                "amount_in":  "₹{:,.2f}",
                "amount_out": "₹{:,.2f}",
            }),
            use_container_width=True,
            height=400,
        )

        st.download_button(
            "⬇️ Download Feature CSV",
            feats_df.to_csv(index=False).encode(),
            "graph_features.csv", "text/csv",
        )

        # Distributions
        st.markdown('<p class="section-hdr">Feature Distributions</p>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Degree distribution**")
            hist_deg = feats_df["degree"].value_counts().sort_index()
            st.bar_chart(hist_deg)
        with col_b:
            st.markdown("**Amount Out distribution (log-binned)**")
            bins = np.logspace(
                np.log10(max(feats_df["amount_out"].min(), 1)),
                np.log10(max(feats_df["amount_out"].max(), 2)),
                15,
            )
            counts, edges = np.histogram(feats_df["amount_out"], bins=bins)
            hist_df = pd.DataFrame({"amount_out_bin": edges[:-1].round(0), "count": counts})
            st.bar_chart(hist_df.set_index("amount_out_bin"))


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICTIONS
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-hdr">Model Predictions</p>', unsafe_allow_html=True)

    feats_df = st.session_state.graph_features
    preds    = st.session_state.predictions

    if not preds or feats_df.empty:
        st.info("Run inference first (Manual Entry or Bulk tab).")
    else:
        result_df = feats_df.copy()
        result_df["fraud_score"] = preds
        result_df["label"] = result_df["fraud_score"].apply(
            lambda s: fraud_label(s)[0]
        )

        n_fraud  = int((result_df["fraud_score"] >= 0.7).sum())
        n_sus    = int(((result_df["fraud_score"] >= 0.4) & (result_df["fraud_score"] < 0.7)).sum())
        n_legit  = int((result_df["fraud_score"] < 0.4).sum())

        ma, mb, mc = st.columns(3)
        ma.metric("🚨 Fraud",    n_fraud,  delta=f"{n_fraud/len(preds)*100:.1f}%", delta_color="inverse")
        mb.metric("⚠️ Suspect",  n_sus)
        mc.metric("✅ Legitimate", n_legit, delta=f"{n_legit/len(preds)*100:.1f}%")

        # Risk gauge bar
        st.markdown("**Risk Score Distribution**")
        fraud_pct = n_fraud / len(preds)
        bar_html  = f"""
        <div style="background:#1e3a5f;border-radius:8px;height:22px;width:100%;overflow:hidden;margin-bottom:18px">
            <div style="height:100%;width:{n_legit/len(preds)*100:.1f}%;background:#00ff88;float:left"></div>
            <div style="height:100%;width:{n_sus/len(preds)*100:.1f}%;background:#ffaa00;float:left"></div>
            <div style="height:100%;width:{n_fraud/len(preds)*100:.1f}%;background:#ff4444;float:left"></div>
        </div>
        <span style="color:#00ff88;font-size:0.75rem">■ Legit</span>&nbsp;&nbsp;
        <span style="color:#ffaa00;font-size:0.75rem">■ Suspect</span>&nbsp;&nbsp;
        <span style="color:#ff4444;font-size:0.75rem">■ Fraud</span>
        """
        st.markdown(bar_html, unsafe_allow_html=True)

        # Sorted full table
        st.markdown('<p class="section-hdr">All Accounts — Sorted by Risk</p>', unsafe_allow_html=True)
        disp = result_df.sort_values("fraud_score", ascending=False).reset_index(drop=True)
        st.dataframe(
            disp.style.background_gradient(subset=["fraud_score"], cmap="RdYlGn_r")
                      .format({"fraud_score": "{:.4f}", "amount_in": "₹{:,.2f}", "amount_out": "₹{:,.2f}"}),
            use_container_width=True,
            height=420,
        )

        # Top fraud cards
        st.markdown('<p class="section-hdr">Top Fraud Alerts</p>', unsafe_allow_html=True)
        top_fraud = disp[disp["fraud_score"] >= 0.7].head(10)
        if top_fraud.empty:
            st.success("No accounts flagged as high-risk 🎉")
        else:
            for _, row in top_fraud.iterrows():
                label, css_cls = fraud_label(row["fraud_score"])
                st.markdown(f"""
                <div class="{css_cls}">
                    <b style="font-family:'Space Mono',monospace;font-size:0.95rem">{row['id']}</b>
                    &nbsp;&nbsp;<span style="opacity:0.7">{label}</span>
                    &nbsp;&nbsp;<b>Score: {row['fraud_score']:.4f}</b><br>
                    <span style="font-size:0.78rem;color:#94a3b8">
                    Degree: {row['degree']} &nbsp;|&nbsp;
                    TXs in: {row['tx_count_in']} &nbsp;|&nbsp;
                    TXs out: {row['tx_count_out']} &nbsp;|&nbsp;
                    ₹In: {row['amount_in']:,.0f} &nbsp;|&nbsp;
                    ₹Out: {row['amount_out']:,.0f} &nbsp;|&nbsp;
                    Cluster: {row['cluster_size']} &nbsp;|&nbsp;
                    Triangles: {row['triangle_count']}
                    </span>
                </div>
                """, unsafe_allow_html=True)

        st.download_button(
            "⬇️ Export Predictions CSV",
            disp.to_csv(index=False).encode(),
            "fraud_predictions.csv", "text/csv",
        )

# ─── ACTIVITY LOG ───────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-hdr">Activity Log</p>', unsafe_allow_html=True)
log_text = "\n".join(st.session_state.log_lines[-40:]) or "No activity yet."
st.markdown(f'<div class="terminal">{log_text}</div>', unsafe_allow_html=True)
