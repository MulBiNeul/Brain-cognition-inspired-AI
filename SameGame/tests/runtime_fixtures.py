"""Small case catalogs and provider responses for offline runner regressions."""

import json


def problem(identifier, rounds=1):
    return {"id": identifier, "type": "same_letter", "rounds": [
        {"round": number, "forbidden": ("AA", "BB", "CC", "DD", "EE")[(number-1) % 5],
         "cases": [{"id": f"c{target}", "input": "A"*20 + "BBCCDDEE" + "CDE"[:target],
                    "target": target} for target in range(4)]}
        for number in range(1, rounds+1)]}


def item(rounds=1):
    return problem("transport-test", rounds)


def answer(request):
    history = request.get("input", request.get("messages"))
    text = json.loads(history[-1]["content"].removeprefix("Initial string: "))
    return json.dumps({"moves": "A1 B1 C1 D1 E1", "answer": text[28:]})


def response(text='{"moves":"A1 B1 C1 D1 E1","answer":""}', *, marker="opaque", status="completed"):
    return {
        "status": status,
        "output": [
            {"type": "reasoning", "encrypted_content": marker, "summary": []},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]},
        ],
        "usage": {"input_tokens": 12, "output_tokens": 8},
    }


def complete_response(request, marker="original-state"):
    return response(answer(request), marker=marker)
