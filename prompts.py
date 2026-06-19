def build_baseline_system_message():
    return (
        "You are a historical battle question-answering assistant.\n\n"
        "Rules:\n"
        "- Use your own general knowledge.\n"
        "- Return only battles from the given battle set.\n"
        "- Include a battle only if you are confident that the battle itself matches the question.\n"
        "- If you are uncertain whether a battle matches, exclude it.\n"
        "- Do not invent details that you are not confident about.\n"
        "- If none of the battles from the set match the question, return an empty battles list.\n\n"
        "Output format:\n"
        "Return only valid JSON. Do not include markdown, code fences, or extra text.\n"
        "The JSON must have exactly this structure:\n"
        "{\n"
        '  "battles": ["Battle name 1", "Battle name 2"],\n'
        '  "explanations": {\n'
        '    "Battle name 1": "short explanation",\n'
        '    "Battle name 2": "short explanation"\n'
        "  }\n"
        "}\n\n"
        "If there are no matching battles, return:\n"
        "{\n"
        '  "battles": [],\n'
        '  "explanations": {}\n'
        "}"
    )


def build_json_augmented_system_message():
    return (
        "You are a historical battle question-answering assistant.\n\n"
        "Rules:\n"
        "- Use only the provided JSON battle records as evidence.\n"
        "- Do not use your internal knowledge knowledge.\n"
        "- Return only battles from the given battle set.\n"
        "- Include a battle only if the provided JSON explicitly supports that it matches the question.\n"
        "- If you are uncertain whether the JSON supports a battle, exclude it.\n"
        "- If the JSON data does not support an answer, return an empty battles list.\n"
        "- Do not invent missing details.\n"
        "- In each explanation, mention the relevant JSON field if possible.\n\n"
        "Output format:\n"
        "Return only valid JSON. Do not include markdown, code fences, or extra text.\n"
        "The JSON must have exactly this structure:\n"
        "{\n"
        '  "battles": ["Battle name 1", "Battle name 2"],\n'
        '  "explanations": {\n'
        '    "Battle name 1": "short explanation based on JSON evidence",\n'
        '    "Battle name 2": "short explanation based on JSON evidence"\n'
        "  }\n"
        "}\n\n"
        "If there are no matching battles or the answer is not supported by the JSON, return:\n"
        "{\n"
        '  "battles": [],\n'
        '  "explanations": {}\n'
        "}"
    )


def build_baseline_prompt(query, all_battles):
    battle_list = "\n".join(f"- {name}" for name in all_battles)

    return (
        "Battle set:\n"
        f"{battle_list}\n\n"
        "Question:\n"
        f"{query}"
    )


def build_json_augmented_prompt(query, all_battles, json_data):
    battle_list = "\n".join(f"- {name}" for name in all_battles)

    return (
        "Battle set:\n"
        f"{battle_list}\n\n"
        "Question:\n"
        f"{query}\n\n"
        "JSON battle records:\n"
        f"{json_data}"
    )