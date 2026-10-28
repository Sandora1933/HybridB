import json
import re
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt

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


CONTEXT_RELEVANCE_MAX_NEW_TOKENS = 1200

CONTEXT_RELEVANCE_RESULTS_DIR = Path(
    "results/e2e_rag/context_relevance"
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


def _parse_context_relevance_judgement(
    response: str,
) -> dict[str, Any]:
    """
    Parse the relevance judgement returned by the evaluator LLM.
    """
    judgement = _parse_json_object(response)

    relevant = judgement.get("relevant")

    if isinstance(relevant, str):
        normalized = relevant.strip().lower()

        if normalized in {"yes", "true", "relevant"}:
            relevant = True
        elif normalized in {
            "no",
            "false",
            "irrelevant",
        }:
            relevant = False

    if not isinstance(relevant, bool):
        raise ValueError(
            "Context-relevance judge did not return a boolean "
            "'relevant' value."
        )

    reason = judgement.get("reason", "")

    if not isinstance(reason, str):
        reason = str(reason)

    return {
        "relevant": relevant,
        "reason": reason.strip(),
    }


def _judge_context_record_relevance(
    question: str,
    context_record: dict[str, Any],
    llm,
) -> dict[str, Any]:
    """
    Decide whether one retrieved JSON battle record is relevant to the
    question.
    """
    qid = context_record.get("qid")
    battle_name = context_record.get("battle_name")

    system_message = (
        "You are evaluating the relevance of a retrieved JSON battle "
        "record to a historical question.\n\n"
        "Determine whether the record contains information that could "
        "directly help answer the question.\n\n"
        "Rules:\n"
        "- Use only the question and the supplied JSON record.\n"
        "- Do not use outside historical knowledge.\n"
        "- Mark the record as relevant only when it contains evidence "
        "that the battle could satisfy the substantive conditions in "
        "the question or directly support an answer explanation.\n"
        "- A record is not relevant merely because it makes it possible "
        "to reject the battle.\n"
        "- Superficial overlap in words is not enough.\n"
        "- If the record concerns a different date, location, conflict, "
        "participant, outcome, tactic, or other central condition, mark "
        "it as irrelevant.\n"
        "- For a multi-condition question, consider all important "
        "conditions together.\n\n"
        "Return only valid JSON with exactly this structure:\n"
        "{\n"
        '  "relevant": true,\n'
        '  "reason": "short explanation"\n'
        "}\n"
        "Keep the reason under 40 words.\n"
        "Stop immediately after the closing JSON brace."
    )

    prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved battle QID:\n{qid}\n\n"
        f"Retrieved battle name:\n{battle_name}\n\n"
        "Retrieved JSON battle record:\n"
        f"{json.dumps(context_record, ensure_ascii=False, indent=2)}"
    )

    last_error = None

    for attempt in range(2):
        try:
            response = generate_answer(
                llm=llm,
                prompt=prompt,
                system_message=system_message,
                max_new_tokens=CONTEXT_RELEVANCE_MAX_NEW_TOKENS,
            )

            judgement = (
                _parse_context_relevance_judgement(response)
            )

            judgement["judge_response"] = response
            return judgement

        except Exception as error:
            last_error = error

            if attempt == 0:
                print(
                    f"      Invalid relevance response for "
                    f"{qid}; retrying..."
                )

    raise RuntimeError(
        f"Could not evaluate context record {qid}: {last_error}"
    )


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


def _calculate_context_relevance_for_file(
    file: str | Path,
    llm,
) -> dict[str, Any]:
    """
    Calculate record-level context relevance for one query.

    Every retrieved JSON battle schema is treated as one context unit.

    Context relevance:
        relevant JSON records
        --------------------- * 100
        all JSON records
    """
    file = Path(file)

    if not file.exists():
        raise FileNotFoundError(
            f"Experiment file not found: {file}"
        )

    with file.open("r", encoding="utf-8") as input_file:
        experiment_result = json.load(input_file)

    query_id = experiment_result.get(
        "query_id",
        file.stem.replace("_answer", ""),
    )

    question = (
        experiment_result.get("generation_query")
        or experiment_result.get("query")
    )

    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            f"Missing query in experiment file: {file}"
        )

    prompt = experiment_result.get("prompt")

    if not isinstance(prompt, str) or not prompt.strip():
        return {
            "query_id": query_id,
            "file": str(file),
            "question": question,
            "status": "skipped_missing_context",
            "context_relevance_percentage": None,
            "relevant_records": 0,
            "total_records": 0,
            "records": [],
        }

    context_records = _extract_json_context(prompt)

    if not context_records:
        return {
            "query_id": query_id,
            "file": str(file),
            "question": question,
            "status": "no_context_records",
            "context_relevance_percentage": None,
            "relevant_records": 0,
            "total_records": 0,
            "records": [],
        }

    record_results = []

    for position, context_record in enumerate(
        context_records,
        start=1,
    ):
        if not isinstance(context_record, dict):
            continue

        qid = context_record.get("qid")
        battle_name = context_record.get("battle_name")
        retrieval_rank = context_record.get(
            "retrieval_rank",
            position,
        )

        print(
            f"      Record {position}/{len(context_records)}: "
            f"{qid} — {battle_name}"
        )

        judgement = _judge_context_record_relevance(
            question=question,
            context_record=context_record,
            llm=llm,
        )

        record_results.append({
            "retrieval_rank": retrieval_rank,
            "qid": qid,
            "battle_name": battle_name,
            "relevant": judgement["relevant"],
            "reason": judgement["reason"],
            "judge_response": judgement["judge_response"],
        })

    total_records = len(record_results)

    relevant_records = sum(
        record["relevant"]
        for record in record_results
    )

    context_relevance_percentage = (
        100.0 * relevant_records / total_records
        if total_records > 0
        else None
    )

    return {
        "query_id": query_id,
        "file": str(file),
        "question": question,
        "status": (
            "evaluated"
            if context_relevance_percentage is not None
            else "no_context_records"
        ),
        "context_relevance_percentage": (
            context_relevance_percentage
        ),
        "relevant_records": relevant_records,
        "total_records": total_records,
        "records": record_results,
    }


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

def calculate_context_relevance(
    top_k: int,
) -> dict[str, Any]:
    """
    Calculate record-level context relevance for a JSON top-k experiment.

    Expected source directories:
        results/e2e_rag/top5_json_details
        results/e2e_rag/top10_json_details
        results/e2e_rag/top20_json_details

    Output:
        results/e2e_rag/context_relevance/
            json_top_<k>_context_relevance_report.json

    The context relevance percentage for each query is:

        100 * relevant records / all retrieved records
    """
    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k <= 0
    ):
        raise ValueError(
            "top_k must be a positive integer."
        )

    source_dir = Path(
        f"results/e2e_rag/top{top_k}_json_details"
    )

    output_file = (
        CONTEXT_RELEVANCE_RESULTS_DIR
        / f"json_top_{top_k}_context_relevance_report.json"
    )

    if not source_dir.exists():
        raise FileNotFoundError(
            f"JSON details directory not found: {source_dir}"
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

    llm = _get_faithfulness_llm()
    query_reports = []

    for file_index, result_file in enumerate(
        result_files,
        start=1,
    ):
        print(
            f"\n[{file_index}/{len(result_files)}] "
            f"Evaluating {result_file.name}"
        )

        try:
            query_report = (
                _calculate_context_relevance_for_file(
                    file=result_file,
                    llm=llm,
                )
            )

        except Exception as error:
            query_report = {
                "query_id": result_file.stem.replace(
                    "_answer",
                    "",
                ),
                "file": str(result_file),
                "status": "evaluation_error",
                "context_relevance_percentage": None,
                "relevant_records": 0,
                "total_records": 0,
                "evaluation_error": str(error),
                "records": [],
            }

            print(f"    Evaluation error: {error}")

        query_reports.append(query_report)

        evaluated_reports = [
            report
            for report in query_reports
            if (
                report.get("context_relevance_percentage")
                is not None
                and report.get("status") == "evaluated"
            )
        ]

        average_context_relevance_percentage = (
            sum(
                report["context_relevance_percentage"]
                for report in evaluated_reports
            )
            / len(evaluated_reports)
            if evaluated_reports
            else None
        )

        total_relevant_records = sum(
            report["relevant_records"]
            for report in evaluated_reports
        )

        total_context_records = sum(
            report["total_records"]
            for report in evaluated_reports
        )

        micro_context_relevance_percentage = (
            100.0
            * total_relevant_records
            / total_context_records
            if total_context_records > 0
            else None
        )

        final_report = {
            "metric": "record_level_context_relevance",
            "framework": "RAGAS-inspired",
            "score_unit": "percentage",
            "score_range": "0 to 100",
            "top_k": top_k,
            "source_directory": str(source_dir),
            "context_unit": (
                "one retrieved JSON battle record"
            ),
            "formula": (
                "100 * relevant_records / total_records"
            ),
            "queries": query_reports,
            "Final_results": {
                "average_context_relevance_percentage": (
                    average_context_relevance_percentage
                ),
                "micro_context_relevance_percentage": (
                    micro_context_relevance_percentage
                ),
                "num_files": len(result_files),
                "num_processed_files": len(
                    query_reports
                ),
                "num_evaluated_queries": len(
                    evaluated_reports
                ),
                "num_skipped_queries": sum(
                    report.get(
                        "status",
                        "",
                    ).startswith("skipped")
                    or report.get("status")
                    == "no_context_records"
                    for report in query_reports
                ),
                "evaluation_error_count": sum(
                    report.get("status")
                    == "evaluation_error"
                    for report in query_reports
                ),
                "total_relevant_records": (
                    total_relevant_records
                ),
                "total_context_records": (
                    total_context_records
                ),
            },
        }

        # Save progressively after every query.
        _save_json(final_report, output_file)

        if (
            query_report.get(
                "context_relevance_percentage"
            )
            is not None
        ):
            print(
                "    Context relevance: "
                f"{query_report['context_relevance_percentage']:.2f}% "
                f"({query_report['relevant_records']}/"
                f"{query_report['total_records']} records)"
            )

    print("\n" + "=" * 80)
    print("CONTEXT RELEVANCE EVALUATION FINISHED")
    print("=" * 80)

    final_results = final_report["Final_results"]

    if (
        final_results[
            "average_context_relevance_percentage"
        ]
        is not None
    ):
        print(
            "Average context relevance: "
            f"{final_results['average_context_relevance_percentage']:.2f}%"
        )

        print(
            "Micro context relevance: "
            f"{final_results['micro_context_relevance_percentage']:.2f}%"
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


def plot_context_relevance_top_k() -> None:
    """
    Plot the relationship between the number of retrieved JSON battle
    records and average context relevance.
    """
    top_k_values = [5, 10, 20]
    context_relevance_scores = [61.304, 55.652, 49.456]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        top_k_values,
        context_relevance_scores,
        color="#C44E00",
        marker="o",
        markersize=8,
        linewidth=2.5,
        label="Average context relevance",
    )

    for top_k, score in zip(
        top_k_values,
        context_relevance_scores,
    ):
        ax.annotate(
            f"{score:.3f}%",
            xy=(top_k, score),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

    ax.set_title(
        "JSON context size and average context relevance"
    )
    ax.set_xlabel(
        "Number of retrieved battles (top-k)"
    )
    ax.set_ylabel(
        "Average context relevance (%)"
    )

    ax.set_xticks(top_k_values)
    ax.set_ylim(20, 80)

    ax.grid(
        axis="both",
        linestyle="--",
        alpha=0.35,
    )
    ax.set_axisbelow(True)
    ax.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    #calculate_context_relevance(top_k=20)
    plot_context_relevance_top_k()