"""Prompt templates for LLM-powered anomaly analysis.

Each function builds a structured prompt string for a specific use case.
Templates are designed for the Ollama-hosted model configured in
``base.yaml`` (default ``nvidia/nemotron-3-nano-4b``).
"""

from __future__ import annotations

from typing import Any


def anomaly_report_prompt(
    run_id: str,
    metrics: dict[str, Any],
    scores_summary: dict[str, Any],
) -> str:
    """Build a prompt for generating a natural-language anomaly report.

    Args:
        run_id: Experiment run identifier.
        metrics: Evaluation metrics dict (precision, recall, f1, etc.).
        scores_summary: Score distribution summary (mean, std, p95, etc.).

    Returns:
        Formatted prompt string.
    """
    metrics_block = "\n".join(f"  - {k}: {v}" for k, v in metrics.items())
    scores_block = "\n".join(f"  - {k}: {v}" for k, v in scores_summary.items())

    return f"""You are an anomaly detection expert analysing experiment run '{run_id}'.

## Evaluation Metrics
{metrics_block}

## Anomaly Score Distribution
{scores_block}

Write a concise anomaly analysis report. Include:
1. Summary of detection performance (precision, recall, F1 if available).
2. Interpretation of the anomaly score distribution.
3. Recommendations for improving detection (threshold tuning, model selection).

Keep the report under 300 words. Use plain language suitable for an operations team."""


def model_recommendation_prompt(data_summary: dict[str, Any]) -> str:
    """Build a prompt for recommending anomaly detection models.

    Args:
        data_summary: Dict with keys like ``n_rows``, ``n_features``,
            ``has_periodicity``, ``noise_level``, ``has_labels``.

    Returns:
        Formatted prompt string.
    """
    summary_block = "\n".join(f"  - {k}: {v}" for k, v in data_summary.items())

    models = (
        "zscore, isolation_forest, matrix_profile, "
        "autoencoder, rnn, lstm, gru, lstm_ae, tcn, "
        "vae, gan, tadgan, tranad, deepar, diffusion, "
        "hybrid_ensemble"
    )

    return (
        "You are an anomaly detection expert. "
        "Based on the following dataset characteristics, "
        "recommend the top 3 anomaly detection models "
        f"from this list:\n\n"
        f"Available models: {models}\n\n"
        f"## Dataset Summary\n{summary_block}\n\n"
        "For each recommended model, explain:\n"
        "1. Why it suits this dataset.\n"
        "2. Key hyperparameters to tune.\n"
        "3. Expected strengths and limitations.\n\n"
        "Respond ONLY with valid JSON in this exact format:\n"
        "{\n"
        '  "recommendations": [\n'
        "    {\n"
        '      "model": "<model_name>",\n'
        '      "reason": "<why this model>",\n'
        '      "hyperparameters": "<what to tune>",\n'
        '      "strengths": "<expected strengths>",\n'
        '      "limitations": "<expected limitations>"\n'
        "    }\n"
        "  ]\n"
        "}"
    )


def tool_selection_prompt(
    available_tools: list[dict[str, Any]],
    user_message: str,
) -> str:
    """Build a prompt for the LLM to select MCP tools.

    Args:
        available_tools: List of tool schema dicts, each with ``name``,
            ``description``, and ``parameters``.
        user_message: The user's natural-language request.

    Returns:
        Formatted prompt string.
    """
    tools_block = ""
    for tool in available_tools:
        name = tool.get("name", "unknown")
        desc = tool.get("description", "")
        params = tool.get("parameters", {})
        tools_block += f"\n### {name}\n{desc}\nParameters: {params}\n"

    return (
        "You are a tool-calling assistant for the "
        "Sentinel anomaly detection platform.\n\n"
        "The user has sent a request. Select the "
        "appropriate tool(s) to fulfil it and "
        "provide the required parameters.\n\n"
        f"## Available Tools\n{tools_block}\n\n"
        f"## User Request\n{user_message}\n\n"
        "Respond ONLY with valid JSON in this "
        "exact format:\n"
        "{\n"
        '  "tool_calls": [\n'
        "    {\n"
        '      "tool": "<tool_name>",\n'
        '      "parameters": {<key>: <value>}\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "If no tool is appropriate, respond with:\n"
        "{\n"
        '  "tool_calls": [],\n'
        '  "message": "<explain why no tool matches>"\n'
        "}"
    )


def data_summary_prompt(dataset_info: dict[str, Any]) -> str:
    """Build a prompt for generating a natural-language data summary.

    Args:
        dataset_info: Dict with dataset metadata (shape, features,
            time_range, stats, etc.).

    Returns:
        Formatted prompt string.
    """
    info_block = "\n".join(f"  - {k}: {v}" for k, v in dataset_info.items())

    return (
        "You are a data analyst. Summarise the "
        "following dataset for an anomaly detection "
        "use case.\n\n"
        f"## Dataset Information\n{info_block}\n\n"
        "Write a concise summary (under 150 words) "
        "covering:\n"
        "1. Dataset size and feature count.\n"
        "2. Time range and temporal resolution "
        "(if determinable).\n"
        "3. Notable characteristics relevant to "
        "anomaly detection.\n"
        "4. Potential preprocessing recommendations."
    )
