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


def build_json_generation_prompt(
    qid: str,
    battle_text: str,
    empty_schema: dict[str, Any],
) -> str:
    """v1. Build a prompt for extracting a historical battle into JSON."""

    schema_text = json.dumps(
        empty_schema,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
You are extracting structured information about one historical battle.

TARGET BATTLE QID:
{qid}

TASK:
Complete the JSON template using the supplied source text.

Extract as much source-supported information as possible for all relevant
schema fields, including formations, commanders, tactics, equipment details,
numerical estimates, subunit casualty figures, battle phases, turning points,
and uncertainty notes.

Do not invent information, use external knowledge, or make unsupported
assumptions.

RULES:

1. Use only information supported by the source text. Do not use external or
   prior knowledge.

2. Extract information only about the target battle and its directly relevant
   background, course, outcome, and consequences. Do not assign facts from
   other battles, unrelated campaign events, or later historical developments
   to the target battle unless the source explicitly connects them to it.

3. Preserve the exact JSON structure, field names, nesting, data types, and
   array item types. Do not add, remove, rename, or restructure fields.

4. Fill as many fields as the source supports.
   - Unsupported scalar fields must be null.
   - Unsupported arrays must be [].
   - Do not return placeholder array objects containing only null or empty
     values.
   - Do not fill a field merely to make the JSON appear complete.

5. Set battle_id and extraction_metadata.wikidata_id to the provided QID.

6. Preserve uncertainty, ranges, approximations, alternative estimates, and
   conflicting figures. Do not replace them with unsupported exact values.
   Use label, note, and uncertainty_notes where appropriate.

7. Do not calculate totals unless all components are explicit, complete,
   unambiguous, and refer to the same side in the target battle. Explain any
   calculation in the corresponding note field.

8. A participant strength total must represent the complete supported strength
   of that side. Do not use the number of one component, such as infantry, as
   the side's total strength when additional forces are also mentioned.

9. Include commanders only when the source clearly connects them to command
   during the target battle. Preserve major allies and supporting contingents
   that actually participated.

10. Keep target-battle casualties separate from casualties belonging to other
    battles, the wider campaign, pursuit, retreat, or later aftermath.

11. Preserve supported subunit casualty figures, but clearly associate them
    with the relevant formation or event. Do not present subunit casualties as
    the total casualties of an entire participant side.

12. Distinguish equipment used in the battle from captured materiel. Do not
    treat captured equipment as pre-battle strength.

13. Concise source-grounded abstraction is allowed for military facets,
    narrative events, turning points, narrative patterns, similarity tags, and
    retrieval summaries. Do not invent unsupported facts, motives, causal
    claims, or specialized terminology.

14. When the source describes a clear battle sequence, populate chronological
    narrative events, important turning points, action sequence, and retrieval
    summaries. Focus on developments important to the overall battle while
    preserving significant supported details.

15. Dates:
    - Use YYYY-MM-DD for CE iso_date when day, month, and year are known.
    - Populate duration_days only when both boundary dates are precise.
    - Count duration inclusively.
    - Do not create non-standard ISO strings for BCE dates.
    - Record important date inconsistencies in uncertainty_notes.

16. Set all confidence fields to null.

17. Return exactly one valid JSON object. Do not include Markdown, comments,
    explanations, or text outside the JSON.

Before returning, silently verify that:
- every populated value is supported by the source;
- no fact from another battle was assigned to the target battle;
- participant totals do not represent only partial forces;
- subunit casualties are not presented as whole-side casualties;
- uncertainty and conflicting estimates are preserved;
- all schema types and array structures are unchanged;
- no empty placeholder objects remain;
- the output is valid JSON.

EMPTY JSON TEMPLATE:

{schema_text}

SOURCE TEXT:

{battle_text}

Return the completed JSON object.
""".strip()