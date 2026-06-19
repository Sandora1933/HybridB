
import json

def evaluate_model_answer(answer, gold_battles):
    parsed_answer = json.loads(answer)
    model_battles = parsed_answer["battles"]

    precision = calculate_precision(
        golden_battles=gold_battles,
        model_battles=model_battles
    )

    recall = calculate_recall(
        golden_battles=gold_battles,
        model_battles=model_battles
    )

    f1 = calculate_f1_score(
        golden_battles=gold_battles,
        model_battles=model_battles
    )

    exact_match = calculate_exact_match(
        golden_battles=gold_battles,
        model_battles=model_battles
    )

    return {
        "model_battles": model_battles,
        "gold_battles": gold_battles,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match
    }


def normalize_battle_name(name):
    return str(name).strip().lower()


def normalize_battle_set(battles):
    return {
        normalize_battle_name(battle)
        for battle in battles
        if battle is not None and str(battle).strip()
    }


def calculate_precision(golden_battles, model_battles):
    golden_set = normalize_battle_set(golden_battles)
    model_set = normalize_battle_set(model_battles)

    if len(model_set) == 0:
        return 0.0

    true_positives = len(golden_set.intersection(model_set))

    return true_positives / len(model_set)


def calculate_recall(golden_battles, model_battles):
    golden_set = normalize_battle_set(golden_battles)
    model_set = normalize_battle_set(model_battles)

    if len(golden_set) == 0:
        return 1.0 if len(model_set) == 0 else 0.0

    true_positives = len(golden_set.intersection(model_set))

    return true_positives / len(golden_set)


def calculate_f1_score(golden_battles, model_battles):
    precision = calculate_precision(golden_battles, model_battles)
    recall = calculate_recall(golden_battles, model_battles)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def calculate_exact_match(golden_battles, model_battles):
    golden_set = normalize_battle_set(golden_battles)
    model_set = normalize_battle_set(model_battles)

    return golden_set == model_set