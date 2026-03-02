"""
CyberFin Nexus — Gemini AI Summarization
Generates natural language explanations for why an account is flagged
as suspicious, using gathered context from the graph, transactions,
cyber events, and model attention weights.
"""

from google import genai
import numpy as np
import time

# Available Gemini models (ordered by preference)
AVAILABLE_MODELS = [
    "gemma-3-1b-it",
    "gemini-3-flash-preview",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]


def gather_account_context(account_id, graph, scores, data_dict, attention_scores=None):
    """
    Collect all available signals for a given account to build
    a comprehensive context string for Gemini.

    Returns:
        dict with all account context, or None if account not found.
    """
    node_map = graph.node_map
    if account_id not in node_map:
        return None

    idx = node_map[account_id]
    accounts_df = data_dict["accounts"]
    transactions_df = data_dict["transactions"]
    cyber_events_df = data_dict["cyber_events"]
    mule_rings_df = data_dict["mule_rings"]

    # --- Basic account info ---
    acc_row = accounts_df[accounts_df.account_id == account_id]
    if len(acc_row) == 0:
        return None
    acc_row = acc_row.iloc[0]

    # --- Risk & attention ---
    risk_score = float(scores[idx])
    attn = float(attention_scores[idx]) if attention_scores is not None and idx < len(attention_scores) else None

    # --- Node features (raw, from graph) ---
    features = graph.x[idx].numpy()
    feature_names = [
        "Cyber Risk Score", "Balance (norm)", "Account Age (norm)",
        "Tx Velocity (norm)", "Avg Tx Amount (norm)", "Device Count (norm)",
        "VPN Usage Freq", "Cyber Event Count (norm)"
    ]
    feature_dict = {name: round(float(val), 3) for name, val in zip(feature_names, features)}

    # --- Connected neighbors ---
    ei = graph.edge_index.numpy()
    neighbors = []
    for j in range(ei.shape[1]):
        if ei[0, j] == idx and ei[1, j] < len(graph.node_ids):
            nid = graph.node_ids[ei[1, j]]
            ntype = graph.node_types[ei[1, j]]
            n_risk = float(scores[ei[1, j]]) if ei[1, j] < len(scores) else 0
            neighbors.append({"id": nid, "type": ntype, "risk": round(n_risk, 3)})

    # --- Transaction history ---
    outgoing = transactions_df[transactions_df.from_account == account_id]
    incoming = transactions_df[transactions_df.to_account == account_id]
    tx_summary = {
        "outgoing_count": len(outgoing),
        "outgoing_total": round(float(outgoing.amount.sum()), 2) if len(outgoing) > 0 else 0,
        "outgoing_to_external": len(outgoing[outgoing.to_account.str.startswith("EXT_")]),
        "incoming_count": len(incoming),
        "incoming_total": round(float(incoming.amount.sum()), 2) if len(incoming) > 0 else 0,
        "suspicious_txns": int(outgoing.is_suspicious.sum()) if len(outgoing) > 0 else 0,
    }

    # --- Cyber events ---
    acc_events = cyber_events_df[cyber_events_df.account_id == account_id]
    cyber_summary = {
        "total_events": len(acc_events),
        "attack_types": acc_events.attack_type.value_counts().to_dict() if len(acc_events) > 0 else {},
        "avg_severity": round(float(acc_events.severity.mean()), 3) if len(acc_events) > 0 else 0,
        "vpn_events": int(acc_events.is_vpn.sum()) if len(acc_events) > 0 else 0,
    }

    # --- Mule ring membership ---
    ring_membership = mule_rings_df[mule_rings_df.account_id == account_id]
    ring_info = ring_membership.ring_id.tolist() if len(ring_membership) > 0 else []

    return {
        "account_id": account_id,
        "bank_id": int(acc_row.bank_id),
        "is_mule_label": bool(acc_row.is_mule),
        "risk_score": risk_score,
        "attention_weight": attn,
        "account_age_days": int(acc_row.account_age_days),
        "avg_balance": float(acc_row.avg_balance),
        "tx_velocity": float(acc_row.tx_velocity),
        "avg_tx_amount": float(acc_row.avg_tx_amount),
        "features": feature_dict,
        "neighbors": neighbors[:10],  # limit for prompt size
        "transactions": tx_summary,
        "cyber_events": cyber_summary,
        "mule_rings": ring_info,
    }


def _build_prompt(context):
    """Build a structured prompt for Gemini from account context."""

    # Neighbor summary
    device_neighbors = [n for n in context["neighbors"] if n["type"] == "device"]
    account_neighbors = [n for n in context["neighbors"] if n["type"] == "account"]
    external_neighbors = [n for n in context["neighbors"] if n["type"] == "external"]
    high_risk_neighbors = [n for n in context["neighbors"] if n["risk"] > 0.5]

    prompt = f"""You are a financial crime analyst AI assistant for CyberFin Nexus, a money mule detection platform.

Analyze the following account data and explain in 3-4 concise sentences why this account has been flagged as suspicious. Focus on the most important risk indicators. Be specific and reference actual data values. Write from the perspective of an investigation report.

ACCOUNT: {context['account_id']} (Bank {context['bank_id']})
RISK SCORE: {context['risk_score']:.1%}
GAT ATTENTION WEIGHT: {context['attention_weight']:.4f}

ACCOUNT PROFILE:
- Account age: {context['account_age_days']} days
- Average balance: ${context['avg_balance']:,.2f}
- Transaction velocity: {context['tx_velocity']:.1f} txns/day
- Avg transaction amount: ${context['avg_tx_amount']:,.2f}

NETWORK CONNECTIONS:
- Connected to {len(device_neighbors)} device(s), {len(account_neighbors)} account(s), {len(external_neighbors)} external endpoint(s)
- {len(high_risk_neighbors)} high-risk neighbors (risk > 0.5)
- Devices: {', '.join(n['id'] for n in device_neighbors[:3]) or 'none'}

TRANSACTION HISTORY:
- {context['transactions']['outgoing_count']} outgoing transactions totaling ${context['transactions']['outgoing_total']:,.2f}
- {context['transactions']['outgoing_to_external']} transfers to external accounts
- {context['transactions']['incoming_count']} incoming transactions totaling ${context['transactions']['incoming_total']:,.2f}
- {context['transactions']['suspicious_txns']} flagged as suspicious

CYBER EVENTS:
- {context['cyber_events']['total_events']} cyber events detected
- Attack types: {context['cyber_events']['attack_types'] or 'none'}
- Average severity: {context['cyber_events']['avg_severity']}
- {context['cyber_events']['vpn_events']} events via VPN

MULE RING MEMBERSHIP: {', '.join(context['mule_rings']) if context['mule_rings'] else 'Not directly assigned to a known ring'}

Write ONLY the 3-4 sentence analysis. Do not include headers, bullet points, or labels. Start directly with the analysis."""

    return prompt


def generate_suspicion_summary(api_key, context, model="gemini-2.0-flash"):
    """
    Generate a natural language summary explaining why an account is suspicious.

    Args:
        api_key: Gemini API key string
        context: dict from gather_account_context()
        model: Gemini model name to use

    Returns:
        str: Gemini-generated explanation, or error message
    """
    if not api_key or not api_key.strip():
        return "⚠️ Please enter your Gemini API key in the sidebar to enable AI analysis."

    if context is None:
        return "⚠️ Could not gather context for this account."

    try:
        client = genai.Client(api_key=api_key.strip())
        prompt = _build_prompt(context)

        # Retry with exponential backoff for rate limits
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                return response.text.strip()
            except Exception as retry_err:
                err_str = str(retry_err)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    if attempt < max_retries:
                        wait_time = (attempt + 1) * 5
                        time.sleep(wait_time)
                        continue
                    return (
                        f"⏳ Rate limit reached for **{model}**. "
                        f"Try selecting a different model in the sidebar "
                        f"(e.g., `gemini-1.5-flash` or `gemini-2.0-flash-lite`), "
                        f"or wait ~60 seconds and try again."
                    )
                raise  # re-raise non-rate-limit errors

    except Exception as e:
        error_msg = str(e)
        if "API_KEY" in error_msg.upper() or "401" in error_msg or "403" in error_msg:
            return "❌ Invalid Gemini API key. Please check your key in the sidebar."
        return f"❌ Gemini API error: {error_msg}"
