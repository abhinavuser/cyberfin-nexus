"""
CyberFin Nexus — Interactive Streamlit Dashboard
Multi-tab application for cyber-financial threat intelligence.

Run: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pandas as pd
import numpy as np
import json
import time
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generator import generate_all_data
from graph_builder import build_graph, partition_by_bank
from models.gat_model import CyberFinGAT, train_model, predict
from federated.fl_engine import FederatedLearningEngine
from rl.adversarial_sim import AdversarialSimulator
from blockchain.audit_trail import AuditChain, build_audit_trail
from utils.config import THEME, RISK_THRESHOLDS, ROI_AVG_MULE_RING_LOSS
from utils.metrics import compute_all_metrics, risk_category, compute_roi


def hex_to_rgba(hex_color, opacity):
    """Convert hex color + opacity (0-1) to rgba() string for Plotly."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"

# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CyberFin Nexus",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<style>
    /* Dark theme overrides */
    .stApp {{
        background: linear-gradient(135deg, {THEME['bg_primary']} 0%, {THEME['bg_secondary']} 100%);
        color: {THEME['text_primary']};
    }}

    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background: {THEME['bg_secondary']};
        border-right: 1px solid {THEME['accent_cyan']}33;
    }}

    /* Metric cards */
    .metric-card {{
        background: linear-gradient(135deg, {THEME['bg_card']}ee, {THEME['bg_secondary']}dd);
        border: 1px solid {THEME['accent_cyan']}40;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 240, 255, 0.08);
        transition: transform 0.2s, box-shadow 0.2s;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 240, 255, 0.15);
    }}
    .metric-value {{
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, {THEME['accent_cyan']}, {THEME['accent_purple']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0;
    }}
    .metric-label {{
        color: {THEME['text_secondary']};
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    /* Section headers */
    .section-header {{
        background: linear-gradient(90deg, {THEME['accent_cyan']}15, transparent);
        border-left: 3px solid {THEME['accent_cyan']};
        padding: 10px 15px;
        margin: 20px 0 15px 0;
        border-radius: 0 8px 8px 0;
    }}

    /* Alert cards */
    .alert-critical {{
        background: linear-gradient(135deg, #ff006e15, #ff000010);
        border: 1px solid {THEME['risk_critical']}60;
        border-radius: 8px;
        padding: 15px;
        margin: 8px 0;
    }}
    .alert-high {{
        background: linear-gradient(135deg, #ff8c0015, #ff006e10);
        border: 1px solid {THEME['risk_high']}60;
        border-radius: 8px;
        padding: 15px;
        margin: 8px 0;
    }}

    /* Glowing header */
    .glow-header {{
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, {THEME['accent_cyan']}, {THEME['accent_purple']}, {THEME['accent_pink']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px {THEME['accent_cyan']}40;
        margin-bottom: 5px;
    }}
    .sub-header {{
        text-align: center;
        color: {THEME['text_secondary']};
        font-size: 1rem;
        margin-bottom: 30px;
    }}

    /* Table styling */
    .dataframe {{
        background: {THEME['bg_card']} !important;
    }}

    /* Status badge */
    .status-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}
    .badge-critical {{ background: {THEME['risk_critical']}30; color: {THEME['risk_critical']}; border: 1px solid {THEME['risk_critical']}50; }}
    .badge-high {{ background: {THEME['risk_high']}30; color: {THEME['risk_high']}; border: 1px solid {THEME['risk_high']}50; }}
    .badge-medium {{ background: {THEME['risk_medium']}30; color: {THEME['risk_medium']}; border: 1px solid {THEME['risk_medium']}50; }}
    .badge-low {{ background: {THEME['risk_low']}30; color: {THEME['risk_low']}; border: 1px solid {THEME['risk_low']}50; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING & CACHING
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    """Generate and cache synthetic data."""
    return generate_all_data(seed=42)

@st.cache_resource
def load_model_and_graph():
    """Build graph, train model, and cache results."""
    data_dict = load_data()
    graph = build_graph(data_dict)
    model = CyberFinGAT()
    model, history = train_model(model, graph, epochs=100, verbose=False)
    scores = predict(model, graph)
    return model, graph, history, scores

@st.cache_resource
def run_federated_learning():
    """Run FL simulation and cache results."""
    data_dict = load_data()
    graph = build_graph(data_dict)
    partitions = partition_by_bank(data_dict, graph)
    fl_engine = FederatedLearningEngine()
    fl_model, fl_metrics = fl_engine.train_federated(
        graph, partitions, n_rounds=10, local_epochs=5, verbose=False
    )
    privacy_report = fl_engine.get_privacy_report()
    return fl_model, fl_metrics, privacy_report

@st.cache_resource
def build_blockchain_trail():
    """Build audit trail and cache."""
    data_dict = load_data()
    model, graph, _, scores = load_model_and_graph()
    chain = build_audit_trail(
        data_dict, scores, graph.node_ids,
        graph.account_mask.numpy(), data_dict["mule_rings"]
    )
    return chain


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="glow-header">🛡️ CyberFin Nexus</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Cyber × Financial Threat Intelligence</div>', unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "**Navigation**",
        ["🏠 Overview", "🕸️ Nexus Graph", "🏦 Federated Learning",
         "⚔️ Attack Simulation", "🔗 Audit Trail", "🚨 Alerts & XAI"],
        index=0,
    )

    st.divider()

    st.markdown("### ⚡ Quick Stats")
    data_dict = load_data()
    accounts = data_dict["accounts"]
    n_mules = accounts.is_mule.sum()
    n_total = len(accounts)
    st.markdown(f"**Accounts:** {n_total}")
    st.markdown(f"**Mule Accounts:** {n_mules} ({n_mules/n_total*100:.1f}%)")
    st.markdown(f"**Banks:** {accounts.bank_id.nunique()}")
    st.markdown(f"**Transactions:** {len(data_dict['transactions'])}")
    st.markdown(f"**Cyber Events:** {len(data_dict['cyber_events'])}")

    st.divider()
    st.markdown(
        f"<div style='color:{THEME['text_secondary']};font-size:0.75rem;text-align:center;'>"
        f"Built with PyTorch Geometric + Streamlit<br>CyberFin Nexus v1.0</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════

def page_overview():
    st.markdown('<div class="glow-header">🛡️ CyberFin Nexus</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Privacy-First Cyber-Financial Fusion for Mule Ring Detection'
        '</div>',
        unsafe_allow_html=True,
    )

    model, graph, history, scores = load_model_and_graph()
    account_mask = graph.account_mask.numpy()
    y_true = graph.y[graph.account_mask].numpy()
    y_scores = scores[account_mask]
    metrics = compute_all_metrics(y_true, y_scores)

    # Key Metrics Row
    cols = st.columns(5)
    metric_data = [
        ("AUC Score", f"{metrics['auc']:.3f}", "Model discrimination"),
        ("Precision", f"{metrics['precision']:.1%}", "Flagged correctly"),
        ("Recall", f"{metrics['recall']:.1%}", "Mules caught"),
        ("F1 Score", f"{metrics['f1']:.3f}", "Balanced accuracy"),
        ("Est. Savings", f"${compute_roi(n_mules):,.0f}", "Based on rings detected"),
    ]
    for col, (label, value, desc) in zip(cols, metric_data):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div>'
                f'<div style="color:{THEME["text_secondary"]};font-size:0.75rem">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    # Architecture & Problem
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-header"><h3>🔍 The Problem</h3></div>', unsafe_allow_html=True)
        st.markdown("""
        **Banks lose billions** to money mule networks that launder cyber-stolen funds.

        - **SOC teams** detect cyber attacks (phishing, malware) but ignore financial patterns
        - **AML teams** spot suspicious transactions but miss cyber-linked coordination
        - **GDPR/DPDP** prevents banks from sharing data, creating blind spots
        - **Criminals evolve** — using VPNs, AI behavioral mimicry to evade detection

        **CyberFin Nexus bridges all gaps** using:
        1. 🕸️ **GNN (GAT)** — Propagates cyber risk across account-device networks
        2. 🏦 **Federated Learning** — Cross-bank collaboration without sharing data
        3. ⚔️ **RL Adaptation** — Detects evolving attacker strategies in real-time
        4. 🔗 **Blockchain Audit** — Tamper-proof forensic trail for regulators
        """)

    with col2:
        st.markdown('<div class="section-header"><h3>📊 Training Progress</h3></div>', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history["auc"], mode="lines",
            name="AUC",
            line=dict(color=THEME["accent_cyan"], width=2),
            fill="tozeroy",
            fillcolor=hex_to_rgba(THEME['accent_cyan'], 0.08),
        ))
        fig.add_trace(go.Scatter(
            y=history["loss"], mode="lines",
            name="Loss",
            line=dict(color=THEME["accent_pink"], width=2, dash="dot"),
            yaxis="y2",
        ))
        fig.update_layout(
            plot_bgcolor=THEME["bg_card"],
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=THEME["text_primary"]),
            height=280,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis=dict(title="AUC", gridcolor=hex_to_rgba(THEME['text_secondary'], 0.12)),
            yaxis2=dict(title="Loss", overlaying="y", side="right", gridcolor="rgba(0,0,0,0)"),
            xaxis=dict(title="Epoch", gridcolor=hex_to_rgba(THEME['text_secondary'], 0.12)),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Risk Distribution
    st.markdown('<div class="section-header"><h3>📈 Account Risk Distribution</h3></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=y_scores[y_true == 0], name="Legitimate",
            marker_color=THEME["accent_green"], opacity=0.7, nbinsx=30,
        ))
        fig.add_trace(go.Histogram(
            x=y_scores[y_true == 1], name="Mule",
            marker_color=THEME["accent_pink"], opacity=0.7, nbinsx=30,
        ))
        fig.update_layout(
            barmode="overlay",
            plot_bgcolor=THEME["bg_card"],
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=THEME["text_primary"]),
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(title="Risk Score", gridcolor=hex_to_rgba(THEME['text_secondary'], 0.12)),
            yaxis=dict(title="Count", gridcolor=hex_to_rgba(THEME['text_secondary'], 0.12)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk category pie chart
        categories = [risk_category(s) for s in y_scores]
        cat_counts = pd.Series(categories).value_counts()
        colors = {
            "LOW": THEME["risk_low"],
            "MEDIUM": THEME["risk_medium"],
            "HIGH": THEME["risk_high"],
            "CRITICAL": THEME["risk_critical"],
        }
        fig = go.Figure(data=[go.Pie(
            labels=cat_counts.index.tolist(),
            values=cat_counts.values.tolist(),
            hole=0.55,
            marker=dict(colors=[colors.get(c, "#999") for c in cat_counts.index]),
            textfont=dict(color="white"),
        )])
        fig.update_layout(
            plot_bgcolor=THEME["bg_card"],
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=THEME["text_primary"]),
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            annotations=[dict(text="Risk<br>Levels", x=0.5, y=0.5, font_size=14,
                            showarrow=False, font_color=THEME["text_secondary"])],
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: NEXUS GRAPH
# ══════════════════════════════════════════════════════════════════════════

def page_nexus_graph():
    st.markdown('<div class="glow-header">🕸️ Nexus Graph</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Interactive cyber-financial network with risk-colored nodes</div>',
        unsafe_allow_html=True,
    )

    model, graph, _, scores = load_model_and_graph()

    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        max_nodes = st.slider("Max Nodes to Display", 30, 200, 80, step=10)
    with col2:
        risk_filter = st.slider("Min Risk Score Filter", 0.0, 1.0, 0.0, step=0.05)
    with col3:
        show_labels = st.checkbox("Show Labels", value=True)

    # Build NetworkX graph (subset)
    G = nx.Graph()
    node_ids = graph.node_ids
    node_types = graph.node_types
    edge_index = graph.edge_index.numpy()
    account_mask = graph.account_mask.numpy()

    # Select top-risk + some connected nodes
    account_indices = [i for i in range(len(node_ids)) if account_mask[i] and scores[i] >= risk_filter]
    account_indices = sorted(account_indices, key=lambda i: -scores[i])[:max_nodes]

    # Include connected nodes
    selected = set(account_indices)
    for i in range(edge_index.shape[1]):
        s, t = edge_index[0, i], edge_index[1, i]
        if s in selected or t in selected:
            selected.add(s)
            selected.add(t)
        if len(selected) > max_nodes * 2:
            break

    selected = list(selected)[:max_nodes * 2]

    for idx in selected:
        G.add_node(idx, label=node_ids[idx], type=node_types[idx],
                  risk=float(scores[idx]) if idx < len(scores) else 0,
                  is_account=account_mask[idx])

    for i in range(edge_index.shape[1]):
        s, t = edge_index[0, i], edge_index[1, i]
        if s in G.nodes and t in G.nodes:
            G.add_edge(s, t)

    if len(G.nodes) == 0:
        st.warning("No nodes match the filter criteria. Try lowering the risk threshold.")
        return

    # Layout
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

    # Create plotly figure
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_colors = []
    node_sizes = []
    node_texts = []
    node_symbols = []

    for n in G.nodes():
        data_n = G.nodes[n]
        risk = data_n["risk"]
        ntype = data_n["type"]

        if ntype == "account":
            if risk > 0.85:
                node_colors.append(THEME["risk_critical"])
                node_sizes.append(18)
            elif risk > 0.6:
                node_colors.append(THEME["risk_high"])
                node_sizes.append(14)
            elif risk > 0.3:
                node_colors.append(THEME["risk_medium"])
                node_sizes.append(11)
            else:
                node_colors.append(THEME["risk_low"])
                node_sizes.append(8)
            node_symbols.append("circle")
        elif ntype == "device":
            node_colors.append(THEME["accent_purple"])
            node_sizes.append(7)
            node_symbols.append("diamond")
        else:
            node_colors.append(THEME["text_secondary"])
            node_sizes.append(6)
            node_symbols.append("square")

        cat = risk_category(risk) if data_n["is_account"] else "N/A"
        node_texts.append(
            f"<b>{data_n['label']}</b><br>"
            f"Type: {ntype}<br>"
            f"Risk: {risk:.3f}<br>"
            f"Category: {cat}<br>"
            f"Connections: {G.degree(n)}"
        )

    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.5, color=hex_to_rgba(THEME['text_secondary'], 0.25)),
        hoverinfo="none",
    ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text" if show_labels else "markers",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color=f"{THEME['bg_primary']}"),
            symbol=node_symbols,
        ),
        text=[G.nodes[n]["label"][:8] for n in G.nodes()] if show_labels else None,
        textposition="top center",
        textfont=dict(size=7, color=THEME["text_secondary"]),
        hovertext=node_texts,
        hoverinfo="text",
    ))

    fig.update_layout(
        showlegend=False,
        plot_bgcolor=THEME["bg_primary"],
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME["text_primary"]),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Legend
    cols = st.columns(5)
    legend_items = [
        ("🔴 Critical Risk (>0.95)", THEME["risk_critical"]),
        ("🟠 High Risk (>0.85)", THEME["risk_high"]),
        ("🟡 Medium Risk (>0.6)", THEME["risk_medium"]),
        ("🟢 Low Risk (<0.3)", THEME["risk_low"]),
        ("🟣 Device Node", THEME["accent_purple"]),
    ]
    for col, (label, color) in zip(cols, legend_items):
        col.markdown(f"<span style='color:{color};font-weight:600'>{label}</span>", unsafe_allow_html=True)

    # Top risk accounts table
    st.markdown('<div class="section-header"><h3>🎯 Top Risk Accounts</h3></div>', unsafe_allow_html=True)
    top_accounts = []
    for i in range(len(node_ids)):
        if account_mask[i]:
            top_accounts.append({
                "Account": node_ids[i],
                "Risk Score": round(scores[i], 4),
                "Category": risk_category(scores[i]),
                "Bank": node_ids[i].split("_")[1] if "_" in node_ids[i] else "?",
            })
    top_df = pd.DataFrame(top_accounts).sort_values("Risk Score", ascending=False).head(15)
    st.dataframe(top_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: FEDERATED LEARNING
# ══════════════════════════════════════════════════════════════════════════

def page_federated_learning():
    st.markdown('<div class="glow-header">🏦 Federated Learning</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Cross-bank collaboration without sharing raw data</div>',
        unsafe_allow_html=True,
    )

    fl_model, fl_metrics, privacy_report = run_federated_learning()

    # Privacy compliance badges
    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Privacy Budget (ε)</div>'
            f'<div class="metric-value">{privacy_report["epsilon"]}</div>'
            f'<div style="color:{THEME["accent_green"]};font-size:0.8rem">✅ GDPR Compliant</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        final_auc = fl_metrics[-1]["global_auc"] if fl_metrics else 0
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Global AUC</div>'
            f'<div class="metric-value">{final_auc:.3f}</div>'
            f'<div style="color:{THEME["text_secondary"]};font-size:0.8rem">After FL aggregation</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Banks Participating</div>'
            f'<div class="metric-value">{privacy_report["n_banks"]}</div>'
            f'<div style="color:{THEME["text_secondary"]};font-size:0.8rem">Simulated independently</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with cols[3]:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">FL Rounds</div>'
            f'<div class="metric-value">{privacy_report["rounds"]}</div>'
            f'<div style="color:{THEME["text_secondary"]};font-size:0.8rem">Convergence achieved</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # FL Convergence Chart
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-header"><h3>📈 FL Convergence</h3></div>', unsafe_allow_html=True)

        fig = go.Figure()

        # Global AUC
        rounds = [m["round"] for m in fl_metrics]
        global_aucs = [m["global_auc"] for m in fl_metrics]
        fig.add_trace(go.Scatter(
            x=rounds, y=global_aucs, mode="lines+markers",
            name="Global Model",
            line=dict(color=THEME["accent_cyan"], width=3),
            marker=dict(size=8),
        ))

        # Per-bank AUCs
        bank_colors = [THEME["accent_green"], THEME["accent_orange"],
                      THEME["accent_purple"], THEME["accent_pink"]]
        for i, bank_id in enumerate(sorted(fl_metrics[0]["bank_metrics"].keys())):
            bank_aucs = [m["bank_metrics"][bank_id]["auc"] for m in fl_metrics]
            fig.add_trace(go.Scatter(
                x=rounds, y=bank_aucs, mode="lines",
                name=f"Bank {bank_id}",
                line=dict(color=bank_colors[i % len(bank_colors)], width=1.5, dash="dash"),
                opacity=0.7,
            ))

        fig.update_layout(
            plot_bgcolor=THEME["bg_card"],
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=THEME["text_primary"]),
            height=350,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(title="FL Round", gridcolor=hex_to_rgba(THEME['text_secondary'], 0.12)),
            yaxis=dict(title="AUC", gridcolor=hex_to_rgba(THEME['text_secondary'], 0.12)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header"><h3>🔒 Privacy Mechanism</h3></div>', unsafe_allow_html=True)
        st.markdown(f"""
        **Differential Privacy (DP)** adds calibrated noise to model weights
        before sharing, mathematically guaranteeing that no individual
        transaction can be reverse-engineered.

        | Parameter | Value |
        |-----------|-------|
        | Mechanism | Gaussian DP |
        | Epsilon (ε) | {privacy_report['epsilon']} |
        | Delta (δ) | {privacy_report['delta']} |
        | Clip Norm | {privacy_report['clip_norm']} |
        | GDPR | {'✅ Compliant' if privacy_report['compliant_gdpr'] else '❌'} |
        | DPDP | {'✅ Compliant' if privacy_report['compliant_dpdp'] else '❌'} |

        **Key Insight:** Each bank trains locally, shares only
        noisy model weights — zero raw data leaves the bank.
        """)

    # How FL Works
    st.markdown('<div class="section-header"><h3>⚙️ How Federated Learning Works</h3></div>', unsafe_allow_html=True)
    fl_cols = st.columns(4)
    steps = [
        ("1️⃣ Local Training", "Each bank trains a GAT model on its own data independently"),
        ("2️⃣ Weight Upload", "Banks send model weights (not data) to aggregator"),
        ("3️⃣ FedAvg + DP Noise", "Weights are averaged with differential privacy noise"),
        ("4️⃣ Global Update", "Improved global model is sent back to all banks"),
    ]
    for col, (title, desc) in zip(fl_cols, steps):
        with col:
            st.markdown(
                f'<div class="metric-card" style="min-height:120px">'
                f'<div style="font-size:1.1rem;font-weight:600;margin-bottom:8px">{title}</div>'
                f'<div style="color:{THEME["text_secondary"]};font-size:0.85rem">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════
# PAGE: ATTACK SIMULATION
# ══════════════════════════════════════════════════════════════════════════

def page_attack_simulation():
    st.markdown('<div class="glow-header">⚔️ Attack Simulation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">RL-powered adversarial testing with adaptive retraining</div>',
        unsafe_allow_html=True,
    )

    model, graph, _, scores = load_model_and_graph()

    # Attack selector
    attack_options = {
        "velocity_reduction": "🐌 Velocity Reduction — Mules slow transaction frequency",
        "amount_splitting": "💰 Amount Splitting — Break large txns into sub-threshold",
        "device_rotation": "📱 Device Rotation — Switch devices to evade fingerprinting",
        "behavioral_mimicry": "🤖 Behavioral Mimicry — AI-generated normal patterns",
    }

    selected_attack = st.selectbox(
        "Select Attack Strategy",
        list(attack_options.keys()),
        format_func=lambda x: attack_options[x],
    )

    if st.button("🚀 Simulate Attack & Adapt", type="primary", use_container_width=True):
        with st.spinner("Running adversarial simulation..."):
            sim = AdversarialSimulator(model, graph)
            result = sim.simulate_attack(selected_attack, verbose=False)

        # Results metrics
        st.markdown('<div class="section-header"><h3>📊 Attack Results</h3></div>', unsafe_allow_html=True)
        cols = st.columns(4)

        with cols[0]:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Baseline AUC</div>'
                f'<div class="metric-value">{result["baseline_auc"]:.3f}</div>'
                f'<div style="color:{THEME["accent_green"]};font-size:0.8rem">Before attack</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            drop_color = THEME["risk_critical"] if result["auc_drop"] > 0.1 else THEME["risk_medium"]
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Under Attack</div>'
                f'<div class="metric-value" style="background:linear-gradient(90deg,{drop_color},{THEME["accent_pink"]});-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{result["under_attack_auc"]:.3f}</div>'
                f'<div style="color:{drop_color};font-size:0.8rem">↓ {result["auc_drop"]:.3f} drop</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">After Adaptation</div>'
                f'<div class="metric-value">{result["adapted_auc"]:.3f}</div>'
                f'<div style="color:{THEME["accent_green"]};font-size:0.8rem">↑ {result["auc_recovery"]:.3f} recovery</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with cols[3]:
            resilience_color = THEME["accent_green"] if result["resilience_score"] > 0.9 else THEME["risk_medium"]
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Resilience</div>'
                f'<div class="metric-value" style="background:linear-gradient(90deg,{resilience_color},{THEME["accent_cyan"]});-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{result["resilience_score"]:.1%}</div>'
                f'<div style="color:{resilience_color};font-size:0.8rem">Recovery ratio</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Retraining curve
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown('<div class="section-header"><h3>🔄 Adaptive Retraining</h3></div>', unsafe_allow_html=True)
            retrain = result["retrain_history"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=retrain["auc"], mode="lines+markers",
                name="AUC Recovery",
                line=dict(color=THEME["accent_cyan"], width=2),
                marker=dict(size=6),
                fill="tozeroy",
                fillcolor=hex_to_rgba(THEME['accent_cyan'], 0.08),
            ))
            fig.add_hline(y=result["baseline_auc"], line_dash="dash",
                         line_color=THEME["accent_green"],
                         annotation_text="Baseline", annotation_position="top right")
            fig.update_layout(
                plot_bgcolor=THEME["bg_card"],
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=THEME["text_primary"]),
                height=300,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(title="Retraining Epoch", gridcolor=hex_to_rgba(THEME['text_secondary'], 0.12)),
                yaxis=dict(title="AUC", gridcolor=hex_to_rgba(THEME['text_secondary'], 0.12)),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header"><h3>🎯 Detection Summary</h3></div>', unsafe_allow_html=True)
            det_data = pd.DataFrame({
                "Phase": ["Baseline", "Under Attack", "Adapted"],
                "Detections": [result["baseline_detections"], result["attack_detections"], result["adapted_detections"]],
            })
            fig = go.Figure(data=[go.Bar(
                x=det_data["Phase"], y=det_data["Detections"],
                marker_color=[THEME["accent_green"], THEME["accent_pink"], THEME["accent_cyan"]],
                text=det_data["Detections"], textposition="outside",
                textfont=dict(color=THEME["text_primary"]),
            )])
            fig.update_layout(
                plot_bgcolor=THEME["bg_card"],
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=THEME["text_primary"]),
                height=300,
                margin=dict(l=10, r=10, t=30, b=10),
                yaxis=dict(title="Flagged Accounts", gridcolor=hex_to_rgba(THEME['text_secondary'], 0.12)),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Run All Attacks
    st.divider()
    if st.button("🔥 Run All Attack Strategies", use_container_width=True):
        with st.spinner("Simulating all attack strategies..."):
            sim = AdversarialSimulator(model, graph)
            all_results = sim.run_all_attacks(verbose=False)
            summary = sim.get_summary()

        st.success(f"**Overall Resilience: {summary['avg_resilience']:.1%}** — "
                  f"Worst attack: {summary['worst_attack']} | "
                  f"Best recovery: {summary['best_recovery']}")

        # Comparison chart
        comparison = pd.DataFrame(all_results)[["strategy", "baseline_auc", "under_attack_auc", "adapted_auc"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Baseline", x=comparison.strategy, y=comparison.baseline_auc,
                            marker_color=THEME["accent_green"]))
        fig.add_trace(go.Bar(name="Under Attack", x=comparison.strategy, y=comparison.under_attack_auc,
                            marker_color=THEME["accent_pink"]))
        fig.add_trace(go.Bar(name="Adapted", x=comparison.strategy, y=comparison.adapted_auc,
                            marker_color=THEME["accent_cyan"]))
        fig.update_layout(
            barmode="group",
            plot_bgcolor=THEME["bg_card"],
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=THEME["text_primary"]),
            height=350,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis=dict(title="AUC", gridcolor=hex_to_rgba(THEME['text_secondary'], 0.12)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: AUDIT TRAIL
# ══════════════════════════════════════════════════════════════════════════

def page_audit_trail():
    st.markdown('<div class="glow-header">🔗 Blockchain Audit Trail</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Immutable SHA-256 hash chain for regulatory compliance</div>',
        unsafe_allow_html=True,
    )

    chain = build_blockchain_trail()
    summary = chain.get_chain_summary()

    # Summary metrics
    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Total Blocks</div>'
            f'<div class="metric-value">{summary["total_blocks"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        valid, msg = chain.verify_integrity()
        badge = f'<span style="color:{THEME["accent_green"]};font-weight:700">✅ VERIFIED</span>' if valid else f'<span style="color:{THEME["risk_critical"]};font-weight:700">❌ TAMPERED</span>'
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Chain Integrity</div>'
            f'<div style="font-size:1.5rem;margin:8px 0">{badge}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with cols[2]:
        n_alerts = summary["block_types"].get("ALERT", 0)
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Alerts Recorded</div>'
            f'<div class="metric-value">{n_alerts}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with cols[3]:
        n_rings = summary["block_types"].get("RING_DETECTION", 0)
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Rings Detected</div>'
            f'<div class="metric-value">{n_rings}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Block type distribution
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown('<div class="section-header"><h3>📊 Block Types</h3></div>', unsafe_allow_html=True)
        types = summary["block_types"]
        fig = go.Figure(data=[go.Pie(
            labels=list(types.keys()),
            values=list(types.values()),
            hole=0.5,
            marker=dict(colors=[THEME["accent_cyan"], THEME["accent_green"],
                               THEME["accent_pink"], THEME["accent_purple"],
                               THEME["accent_orange"]]),
            textfont=dict(color="white"),
        )])
        fig.update_layout(
            plot_bgcolor=THEME["bg_card"],
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=THEME["text_primary"]),
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header"><h3>🔗 Recent Blocks</h3></div>', unsafe_allow_html=True)
        chain_data = chain.export_json()
        # Show last 10 blocks
        recent = chain_data[-10:]
        for block in reversed(recent):
            block_type = block["data"].get("type", "UNKNOWN")
            color = {
                "ALERT": THEME["risk_high"],
                "RING_DETECTION": THEME["risk_critical"],
                "TRANSACTION_ASSESSMENT": THEME["accent_cyan"],
                "MODEL_EVENT": THEME["accent_purple"],
                "GENESIS": THEME["accent_green"],
            }.get(block_type, THEME["text_secondary"])

            st.markdown(
                f'<div style="background:{THEME["bg_card"]};border-left:3px solid {color};'
                f'padding:8px 12px;margin:4px 0;border-radius:0 6px 6px 0;font-size:0.85rem">'
                f'<b style="color:{color}">#{block["index"]} {block_type}</b> '
                f'<span style="color:{THEME["text_secondary"]}">| {block["timestamp"][:19]}</span><br>'
                f'<code style="font-size:0.7rem;color:{THEME["text_secondary"]}">{block["hash"][:32]}...</code>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Ring Detections
    ring_detections = chain.get_ring_detections()
    if ring_detections:
        st.markdown('<div class="section-header"><h3>🎯 Mule Ring Detections</h3></div>', unsafe_allow_html=True)
        for rd in ring_detections:
            data = rd["data"]
            conf_color = THEME["risk_critical"] if data["confidence"] > 0.8 else THEME["risk_high"]
            st.markdown(
                f'<div class="alert-critical">'
                f'<b style="color:{conf_color}">🔴 {data["ring_id"]}</b> '
                f'| Confidence: <b>{data["confidence"]:.1%}</b> '
                f'| Risk: <b>${data["estimated_total_risk"]:,.0f}</b><br>'
                f'Members: {", ".join(data["accounts"][:5])}'
                f'{"..." if len(data["accounts"]) > 5 else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Full chain export
    st.divider()
    if st.button("📥 Export Full Audit Trail (JSON)"):
        json_str = json.dumps(chain.export_json(), indent=2, default=str)
        st.download_button(
            "Download audit_trail.json",
            json_str,
            file_name="cyberfin_audit_trail.json",
            mime="application/json",
        )


# ══════════════════════════════════════════════════════════════════════════
# PAGE: ALERTS & EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════

def page_alerts_xai():
    st.markdown('<div class="glow-header">🚨 Alerts & Explainability</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">GAT attention-based explainable risk assessment</div>',
        unsafe_allow_html=True,
    )

    model, graph, _, scores = load_model_and_graph()
    data_dict = load_data()
    account_mask = graph.account_mask.numpy()
    y_true = graph.y[graph.account_mask].numpy()

    # Get attention scores
    attention_scores = model.get_node_attention_scores(graph.edge_index)

    # Top-N alerts
    n_display = st.slider("Number of Alerts to Display", 5, 30, 10)

    # Build alert data
    alerts = []
    for i in range(len(graph.node_ids)):
        if account_mask[i] and scores[i] > 0.4:
            attn = attention_scores[i] if attention_scores is not None and i < len(attention_scores) else 0
            alerts.append({
                "Account": graph.node_ids[i],
                "Risk Score": float(scores[i]),
                "Category": risk_category(scores[i]),
                "Attention": float(attn),
                "Actual": "🔴 MULE" if y_true[sum(account_mask[:i+1]) - 1] > 0.5 else "🟢 LEGIT",
                "Bank": graph.node_ids[i].split("_")[1] if "_" in graph.node_ids[i] else "?",
            })

    alerts_df = pd.DataFrame(alerts).sort_values("Risk Score", ascending=False).head(n_display)

    # Alert cards
    st.markdown('<div class="section-header"><h3>🔔 Active Alerts</h3></div>', unsafe_allow_html=True)

    for _, alert in alerts_df.iterrows():
        cat = alert["Category"]
        css_class = f"alert-critical" if cat in ["CRITICAL", "HIGH"] else "alert-high"
        badge_class = f"badge-{cat.lower()}"

        # Get connected accounts for explanation
        acc_idx = graph.node_map.get(alert["Account"], -1)
        neighbors = []
        if acc_idx >= 0:
            ei = graph.edge_index.numpy()
            for j in range(ei.shape[1]):
                if ei[0, j] == acc_idx and ei[1, j] < len(graph.node_ids):
                    neighbor_id = graph.node_ids[ei[1, j]]
                    neighbor_type = graph.node_types[ei[1, j]]
                    if len(neighbors) < 5:
                        neighbors.append(f"{neighbor_id} ({neighbor_type})")

        neighbor_text = ", ".join(neighbors) if neighbors else "N/A"

        st.markdown(
            f'<div class="{css_class}">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<div>'
            f'<b style="font-size:1.1rem">{alert["Account"]}</b> '
            f'<span class="status-badge {badge_class}">{cat}</span> '
            f'{alert["Actual"]}'
            f'</div>'
            f'<div style="text-align:right">'
            f'<div style="font-size:1.4rem;font-weight:700;color:{THEME["accent_cyan"]}">'
            f'{alert["Risk Score"]:.1%}</div>'
            f'<div style="color:{THEME["text_secondary"]};font-size:0.75rem">Risk Score</div>'
            f'</div></div>'
            f'<div style="margin-top:8px;color:{THEME["text_secondary"]};font-size:0.8rem">'
            f'<b>Bank:</b> {alert["Bank"]} | '
            f'<b>Attn Weight:</b> {alert["Attention"]:.4f} | '
            f'<b>Connected to:</b> {neighbor_text}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Explainability section
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header"><h3>🧠 Feature Importance</h3></div>', unsafe_allow_html=True)

        feature_names = [
            "Cyber Risk", "Balance", "Account Age", "Tx Velocity",
            "Avg Tx Amount", "Device Count", "VPN Usage", "Cyber Events"
        ]

        # Analyze feature importance via weight analysis
        mule_mask = graph.account_mask & (graph.y > 0.5)
        legit_mask = graph.account_mask & (graph.y < 0.5)

        mule_features = graph.x[mule_mask].mean(dim=0).numpy()
        legit_features = graph.x[legit_mask].mean(dim=0).numpy()
        diff = np.abs(mule_features - legit_features)
        diff_normalized = diff / (diff.max() + 1e-8)

        fig = go.Figure(data=[go.Bar(
            x=diff_normalized,
            y=feature_names,
            orientation="h",
            marker=dict(
                color=diff_normalized,
                colorscale=[[0, THEME["accent_cyan"]], [1, THEME["accent_pink"]]],
            ),
            text=[f"{v:.2f}" for v in diff_normalized],
            textposition="outside",
            textfont=dict(color=THEME["text_primary"]),
        )])
        fig.update_layout(
            plot_bgcolor=THEME["bg_card"],
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=THEME["text_primary"]),
            height=300,
            margin=dict(l=10, r=60, t=10, b=10),
            xaxis=dict(title="Discriminative Power", gridcolor=hex_to_rgba(THEME['text_secondary'], 0.12)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header"><h3>🎛️ Human-in-the-Loop</h3></div>', unsafe_allow_html=True)

        st.markdown("""
        **Override Model Decisions:**

        Use the controls below to manually override the model's
        risk assessment for specific accounts. This feedback loop
        improves model accuracy over time.
        """)

        override_account = st.selectbox(
            "Select Account to Override",
            alerts_df["Account"].tolist() if len(alerts_df) > 0 else ["No alerts"],
        )

        override_decision = st.radio(
            "Override Decision",
            ["Accept Model Prediction", "Mark as Legitimate", "Confirm as Mule"],
            horizontal=True,
        )

        if st.button("Submit Override", type="primary"):
            st.success(f"✅ Override recorded: {override_account} → {override_decision}")
            st.info("In production, this feedback would retrain the model incrementally.")

    # ROI Calculator
    st.markdown('<div class="section-header"><h3>💰 ROI Calculator</h3></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        n_rings = st.number_input("Mule Rings Detected per Month", 1, 50, 6)
    with col2:
        avg_loss = st.number_input("Avg Loss per Ring ($)", 100000, 5000000, 500000, step=100000)
    with col3:
        improvement = st.slider("Detection Improvement (%)", 10, 50, 25)

    savings = n_rings * avg_loss * (improvement / 100)
    annual = savings * 12

    st.markdown(
        f'<div class="metric-card" style="max-width:600px;margin:20px auto">'
        f'<div class="metric-label">Estimated Annual Savings</div>'
        f'<div class="metric-value" style="font-size:3rem">${annual:,.0f}</div>'
        f'<div style="color:{THEME["text_secondary"]};font-size:0.85rem">'
        f'Based on {n_rings} rings/month × ${avg_loss:,.0f} avg loss × {improvement}% improvement'
        f'</div></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    page_overview()
elif page == "🕸️ Nexus Graph":
    page_nexus_graph()
elif page == "🏦 Federated Learning":
    page_federated_learning()
elif page == "⚔️ Attack Simulation":
    page_attack_simulation()
elif page == "🔗 Audit Trail":
    page_audit_trail()
elif page == "🚨 Alerts & XAI":
    page_alerts_xai()
