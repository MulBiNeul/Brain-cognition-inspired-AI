"""Readable provider summaries for analysis; never participant messages or grading input."""

import hashlib


def summary_artifact(envelope, *, provider, model, call, episode, problem_id,
                     condition, round_number, attempt, trial_path):
    """Copy readable native fields only. Opaque state is never decoded or reconstructed."""
    raw = envelope["response"]
    parts = []
    if provider == "openai":
        for item in raw.get("output") or []:
            if item.get("type") != "reasoning":
                continue
            for block in item.get("summary") or []:
                text = block.get("text")
                if block.get("type") == "summary_text" and isinstance(text, str) and text.strip():
                    parts.append(text)
    elif provider == "anthropic":
        # This does not enable Anthropic display; preserve readable blocks only if returned.
        for block in raw.get("content") or []:
            text = block.get("thinking")
            if block.get("type") == "thinking" and isinstance(text, str) and text.strip():
                parts.append(text)
    status = "available" if parts else "not_returned"
    response_status = raw.get("status") or raw.get("stop_reason") or "unknown"
    metadata = {
        "schema_version": 1,
        "call": call, "episode": episode, "problem_id": problem_id, "condition": condition,
        "round": round_number, "attempt": attempt, "provider": provider, "model": model,
        "response_id": raw.get("id"), "response_status": response_status,
        "transport_attempt": envelope["transport_attempt"],
        "response_sha256": envelope["response_sha256"],
        "request_sha256": envelope["request_sha256"],
        "status": status, "summary_text_count": len(parts),
        "summary_characters": sum(len(text) for text in parts),
        "path": f"{trial_path}/reasoning_summary.txt",
    }
    header = [
        "Reasoning summary — provider-returned text for researcher analysis",
        "This is not a full transcript of internal reasoning.",
        f"Model: {model} ({provider})",
        f"Problem: {problem_id} | Type: {condition} | Round: {round_number} | Attempt: {attempt}",
        f"Call: {call} | Episode: {episode}",
        f"Response ID: {raw.get('id') or 'unknown'}",
        f"Response status: {response_status}",
        f"Summary status: {status}",
        f"Source: transport_{envelope['transport_attempt']:03d}_response.json",
        f"Response SHA256: {envelope['response_sha256']}",
    ]
    if response_status not in ("completed", "end_turn"):
        header.append("The response did not end normally; any returned summary may be incomplete.")
    if parts:
        body = "\n\n".join(f"[Summary {index}]\n{text}" for index, text in enumerate(parts, 1))
    else:
        body = ("No readable reasoning summary was returned in the saved response.\n"
                "This does not mean that the model performed no reasoning.\n"
                "For interrupted streams, any partial events remain in the original stream log.")
    document = "\n".join(header) + "\n\n" + body + "\n"
    metadata["text_sha256"] = hashlib.sha256(document.encode("utf-8")).hexdigest()
    return document, metadata
