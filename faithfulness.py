import json
import re
from pathlib import Path
from typing import Any

from llm import generate_answer, load_llm


JSON_DETAILS_DIR = Path(
    "results/e2e_rag/top20_json_details"
)

FAITHFULNESS_REPORT_FILE = Path(
    "results/e2e_rag/top20_json_faithfulness_report.json"
)

FAITHFULNESS_MAX_NEW_TOKENS = 1200

_faithfulness_llm = None


def _get_faithfulness_llm():
    """
    Load the evaluation LLM once and reuse it for all files.
    """
    global _faithfulness_llm

    if _faithfulness_llm is None:
        print("Loading faithfulness evaluation LLM...")
        _faithfulness_llm = load_llm()

    return _faithfulness_llm


def _save_json(data: dict[str, Any], output_file: str | Path) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def _remove_code_fence(text: str) -> str:
    """
    Remove optional Markdown JSON code fences.
    """
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return text


def _parse_json_object(text: str) -> dict[str, Any]:
    """
    Parse a JSON object returned by a model.

    Also handles answers surrounded by Markdown code fences or small
    amounts of extra text.
    """
    text = _remove_code_fence(text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        object_start = text.find("{")
        object_end = text.rfind("}")

        if object_start == -1 or object_end == -1:
            raise ValueError(
                "Could not find a JSON object in the model response."
            )

        parsed = json.loads(
            text[object_start:object_end + 1]
        )

    if not isinstance(parsed, dict):
        raise ValueError(
            "Expected a JSON object, but received another JSON type."
        )

    return parsed


def _extract_answer_data(
    experiment_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract and parse the model answer from an experiment result.
    """
    answer = experiment_result.get("answer")

    if isinstance(answer, dict):
        return answer

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(
            "The experiment file does not contain a valid model answer."
        )

    return _parse_json_object(answer)


def _extract_json_context(
    prompt: str,
) -> list[dict[str, Any]]:
    """
    Extract the retrieved JSON battle records from the experiment prompt.
    """
    marker = "Retrieved JSON battle records:"

    marker_position = prompt.find(marker)

    if marker_position == -1:
        raise ValueError(
            f"Could not find {marker!r} in the prompt."
        )

    context_text = prompt[
        marker_position + len(marker):
    ].lstrip()

    try:
        context, _ = json.JSONDecoder().raw_decode(context_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Could not parse the retrieved JSON context from the prompt."
        ) from error

    if not isinstance(context, list):
        raise ValueError(
            "Retrieved JSON context must be a list of battle records."
        )

    return context


def _build_context_index(
    context_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Index retrieved records by QID.
    """
    context_index = {}

    for context_item in context_records:
        if not isinstance(context_item, dict):
            continue

        qid = context_item.get("qid")

        if not isinstance(qid, str):
            record = context_item.get("record", {})

            if isinstance(record, dict):
                qid = (
                    record.get("battle_id")
                    or record.get(
                        "extraction_metadata",
                        {},
                    ).get("wikidata_id")
                )

        if isinstance(qid, str):
            context_index[qid] = context_item

    return context_index


def _parse_faithfulness_judgement(
    response: str,
) -> dict[str, Any]:
    """
    Parse the JSON response returned by the faithfulness judge.
    """
    judgement = _parse_json_object(response)

    supported = judgement.get("supported")

    if isinstance(supported, str):
        normalized = supported.strip().lower()

        if normalized in {"yes", "true", "supported"}:
            supported = True
        elif normalized in {
            "no",
            "false",
            "unsupported",
        }:
            supported = False

    if not isinstance(supported, bool):
        raise ValueError(
            "Faithfulness judge did not return a boolean "
            "'supported' value."
        )

    reason = judgement.get("reason", "")

    if not isinstance(reason, str):
        reason = str(reason)

    return {
        "supported": supported,
        "reason": reason.strip(),
    }


def _judge_statement_faithfulness(
    qid: str,
    battle_name: str | None,
    statement: str,
    context_record: dict[str, Any],
    llm,
) -> dict[str, Any]:
    """
    Determine whether one explanation is fully supported by its JSON record.
    """
    system_message = (
        "You are evaluating the faithfulness of a RAG-generated "
        "historical answer.\n\n"
        "Determine whether the statement is fully supported by the "
        "provided JSON battle record.\n\n"
        "Rules:\n"
        "- Use only the provided JSON record.\n"
        "- Do not use your own historical knowledge.\n"
        "- The statement is supported only if every factual claim in it "
        "can be inferred from the JSON record.\n"
        "- If only part of the statement is supported, return false.\n"
        "- A paraphrase does not need to use exactly the same words as "
        "the JSON record.\n"
        "- If the record is missing the required evidence, return false.\n"
        "- Be strict but do not reject reasonable direct inferences.\n\n"
        "Return only valid JSON with exactly this structure:\n"
        "{\n"
        '  "supported": true,\n'
        '  "reason": "short explanation of the judgement"\n'
        "}"
    )

    prompt = (
        f"Battle QID:\n{qid}\n\n"
        f"Battle name:\n{battle_name or 'Unknown'}\n\n"
        f"Statement to verify:\n{statement}\n\n"
        "JSON battle record:\n"
        f"{json.dumps(context_record, ensure_ascii=False, indent=2)}"
    )

    last_error = None

    # Retry once if the evaluator returns invalid JSON.
    for attempt in range(2):
        try:
            response = generate_answer(
                llm=llm,
                prompt=prompt,
                system_message=system_message,
                max_new_tokens=FAITHFULNESS_MAX_NEW_TOKENS,
            )

            judgement = _parse_faithfulness_judgement(response)
            judgement["judge_response"] = response
            return judgement

        except Exception as error:
            last_error = error

            if attempt == 0:
                print(
                    f"    Invalid judge response for {qid}; retrying..."
                )

    raise RuntimeError(
        f"Could not evaluate statement for {qid}: {last_error}"
    )


def calculate_faithfulness(
    file: str | Path,
) -> dict[str, Any]:
    """
    Calculate explanation-level faithfulness for one experiment file.

    One statement is one entry in answer["explanations"].

    Faithfulness:
        number of fully supported explanations
        --------------------------------------
        total number of explanations

    Files with a source run error or no explanations receive no numerical
    faithfulness score and should be excluded from the final average.
    """
    file = Path(file)

    if not file.exists():
        raise FileNotFoundError(
            f"Experiment result file not found: {file}"
        )

    with file.open("r", encoding="utf-8") as input_file:
        experiment_result = json.load(input_file)

    query_id = experiment_result.get(
        "query_id",
        file.stem.replace("_answer", ""),
    )

    source_run_error = (
        experiment_result
        .get("metrics", {})
        .get("run_error")
    )

    if source_run_error is not None:
        return {
            "query_id": query_id,
            "file": str(file),
            "status": "skipped_source_run_error",
            "faithfulness": None,
            "supported_statements": 0,
            "total_statements": 0,
            "source_run_error": source_run_error,
            "statements": [],
        }

    prompt = experiment_result.get("prompt")

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            f"Missing prompt in experiment file: {file}"
        )

    answer_data = _extract_answer_data(experiment_result)
    context_records = _extract_json_context(prompt)
    context_index = _build_context_index(context_records)

    explanations = answer_data.get("explanations", {})
    battle_names = answer_data.get("battle_names", {})

    if not isinstance(explanations, dict):
        raise ValueError(
            f"answer['explanations'] must be an object in: {file}"
        )

    if not isinstance(battle_names, dict):
        battle_names = {}

    if not explanations:
        return {
            "query_id": query_id,
            "file": str(file),
            "status": "no_statements",
            "faithfulness": None,
            "supported_statements": 0,
            "total_statements": 0,
            "statements": [],
        }

    llm = _get_faithfulness_llm()
    statement_results = []

    for qid, explanation in explanations.items():
        if not isinstance(explanation, str):
            explanation = str(explanation)

        explanation = explanation.strip()

        if not explanation:
            continue

        battle_name = battle_names.get(qid)
        context_record = context_index.get(qid)

        print(f"    Evaluating statement for {qid}...")

        # If the generated QID is absent from the retrieved context,
        # its explanation cannot be grounded in that context.
        if context_record is None:
            statement_result = {
                "qid": qid,
                "battle_name": battle_name,
                "statement": explanation,
                "supported": False,
                "reason": (
                    "The battle QID is absent from the retrieved "
                    "JSON context."
                ),
            }
        else:
            judgement = _judge_statement_faithfulness(
                qid=qid,
                battle_name=battle_name,
                statement=explanation,
                context_record=context_record,
                llm=llm,
            )

            statement_result = {
                "qid": qid,
                "battle_name": battle_name,
                "statement": explanation,
                "supported": judgement["supported"],
                "reason": judgement["reason"],
                "judge_response": judgement["judge_response"],
            }

        statement_results.append(statement_result)

    total_statements = len(statement_results)
    supported_statements = sum(
        statement["supported"]
        for statement in statement_results
    )

    faithfulness = (
        supported_statements / total_statements
        if total_statements > 0
        else None
    )

    return {
        "query_id": query_id,
        "file": str(file),
        "status": (
            "evaluated"
            if faithfulness is not None
            else "no_statements"
        ),
        "faithfulness": faithfulness,
        "supported_statements": supported_statements,
        "total_statements": total_statements,
        "statements": statement_results,
    }


def calculate_faithfulness_for_directory(
    source_dir: str | Path = JSON_DETAILS_DIR,
    output_file: str | Path = FAITHFULNESS_REPORT_FILE,
) -> dict[str, Any]:
    """
    Calculate faithfulness for all qXXX_answer.json files.

    Produces:
        - a faithfulness score for every successfully evaluated query;
        - macro-average faithfulness across queries;
        - micro-average faithfulness across all explanations;
        - counts of evaluated, skipped, and failed files.
    """
    source_dir = Path(source_dir)
    output_file = Path(output_file)

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Results directory not found: {source_dir}"
        )

    def query_number(file_path: Path) -> int:
        match = re.search(r"q(\d+)", file_path.stem)

        return int(match.group(1)) if match else 10**9

    result_files = sorted(
        source_dir.glob("q*_answer.json"),
        key=query_number,
    )

    if not result_files:
        raise FileNotFoundError(
            f"No qXXX_answer.json files found in: {source_dir}"
        )

    query_reports = []

    for index, result_file in enumerate(result_files, start=1):
        print(
            f"\n[{index}/{len(result_files)}] "
            f"Evaluating {result_file.name}"
        )

        try:
            query_report = calculate_faithfulness(
                result_file
            )
        except Exception as error:
            query_report = {
                "query_id": result_file.stem.replace(
                    "_answer",
                    "",
                ),
                "file": str(result_file),
                "status": "evaluation_error",
                "faithfulness": None,
                "supported_statements": 0,
                "total_statements": 0,
                "evaluation_error": str(error),
                "statements": [],
            }

            print(f"    Evaluation error: {error}")

        query_reports.append(query_report)

        evaluated_reports = [
            report
            for report in query_reports
            if report["faithfulness"] is not None
        ]

        total_supported = sum(
            report["supported_statements"]
            for report in evaluated_reports
        )
        total_statements = sum(
            report["total_statements"]
            for report in evaluated_reports
        )

        average_faithfulness = (
            sum(
                report["faithfulness"]
                for report in evaluated_reports
            )
            / len(evaluated_reports)
            if evaluated_reports
            else None
        )

        micro_faithfulness = (
            total_supported / total_statements
            if total_statements > 0
            else None
        )

        final_report = {
            "metric": "faithfulness",
            "framework": "RAGAS-inspired",
            "statement_unit": (
                "one complete explanation per predicted battle"
            ),
            "source_directory": str(source_dir),
            "queries": query_reports,
            "Final_results": {
                "average_faithfulness": average_faithfulness,
                "micro_faithfulness": micro_faithfulness,
                "num_files": len(result_files),
                "num_processed_files": len(query_reports),
                "num_evaluated_queries": len(evaluated_reports),
                "num_skipped_queries": sum(
                    report["status"].startswith("skipped")
                    or report["status"] == "no_statements"
                    for report in query_reports
                ),
                "evaluation_error_count": sum(
                    report["status"] == "evaluation_error"
                    for report in query_reports
                ),
                "total_supported_statements": total_supported,
                "total_statements": total_statements,
            },
        }

        # Save progressively so completed evaluations survive interruption.
        _save_json(final_report, output_file)

        if query_report["faithfulness"] is not None:
            print(
                "    Faithfulness: "
                f"{query_report['faithfulness']:.4f} "
                f"({query_report['supported_statements']}/"
                f"{query_report['total_statements']})"
            )

    print("\n" + "=" * 80)
    print("FAITHFULNESS EVALUATION FINISHED")
    print("=" * 80)

    final_results = final_report["Final_results"]

    if final_results["average_faithfulness"] is not None:
        print(
            "Average faithfulness: "
            f"{final_results['average_faithfulness']:.4f}"
        )
        print(
            "Micro faithfulness: "
            f"{final_results['micro_faithfulness']:.4f}"
        )

    print(
        "Evaluated queries: "
        f"{final_results['num_evaluated_queries']}"
    )
    print(
        "Skipped queries: "
        f"{final_results['num_skipped_queries']}"
    )
    print(
        "Evaluation errors: "
        f"{final_results['evaluation_error_count']}"
    )
    print(f"Report saved to: {output_file}")

    return final_report


if __name__ == "__main__":
    calculate_faithfulness_for_directory()