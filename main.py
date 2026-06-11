import json
import os
import requests
from pydantic import BaseModel, Field, ConfigDict


PROPERTY_CACHE_FILE = "wikidata_property_cache.json"
LABEL_CACHE_FILE = "wikidata_label_cache.json"

HEADERS = {
    "User-Agent": "WikidataBattleRetriever/1.0 (vlad8aromanov@gmail.com)"
}


class WikidataEntity(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str | None = None
    labels: dict = Field(default_factory=dict)
    descriptions: dict = Field(default_factory=dict)
    aliases: dict = Field(default_factory=dict)
    claims: dict = Field(default_factory=dict)
    sitelinks: dict = Field(default_factory=dict)


class WikidataPage(BaseModel):
    model_config = ConfigDict(extra="allow")

    entities: dict[str, WikidataEntity]


def load_json_cache(file_path):
    if not os.path.exists(file_path):
        return {}

    if os.path.getsize(file_path) == 0:
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: {file_path} is invalid JSON. Starting with empty cache.")
        return {}


def save_json_cache(cache, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def load_property_cache():
    return load_json_cache(PROPERTY_CACHE_FILE)


def save_property_cache(cache):
    save_json_cache(cache, PROPERTY_CACHE_FILE)


def load_label_cache():
    return load_json_cache(LABEL_CACHE_FILE)


def save_label_cache(cache):
    save_json_cache(cache, LABEL_CACHE_FILE)


def retrieve_entities(ids):
    if not ids:
        return {}

    all_entities = {}
    batch_size = 50

    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        ids_string = "|".join(batch)

        url = (
            "https://www.wikidata.org/w/api.php"
            "?action=wbgetentities"
            "&format=json"
            "&languages=en"
            "&ids=" + ids_string
        )

        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        all_entities.update(response.json()["entities"])

    return all_entities


def update_property_cache(property_ids, language="en"):
    cache = load_property_cache()

    missing = [
        property_id
        for property_id in property_ids
        if property_id not in cache
    ]

    if not missing:
        return cache

    entities = retrieve_entities(missing)

    for property_id, entity in entities.items():
        datatype = entity.get("datatype")

        # Skip identifiers entirely
        if datatype == "external-id":
            continue

        cache[property_id] = {
            "label": entity.get("labels", {})
                           .get(language, {})
                           .get("value", property_id),
            "datatype": datatype
        }

    save_property_cache(cache)

    return cache


def update_label_cache(entity_ids, language="en"):
    cache = load_label_cache()

    missing = [
        entity_id
        for entity_id in entity_ids
        if entity_id not in cache
    ]

    if not missing:
        return cache

    entities = retrieve_entities(missing)

    for entity_id, entity in entities.items():
        cache[entity_id] = {
            "label": entity.get("labels", {})
                           .get(language, {})
                           .get("value", entity_id)
        }

    save_label_cache(cache)

    return cache


def retrieve_wikidata_page_by_id(entity_id):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"

    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    return WikidataPage.model_validate(response.json())


def get_entity(page, entity_id):
    return page.entities[entity_id]


def collect_referenced_entity_ids(entity):
    entity_ids = set()

    for claims in entity.claims.values():
        for claim in claims:
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue")

            if not datavalue:
                continue

            value = datavalue.get("value")

            if isinstance(value, dict) and "id" in value:
                entity_ids.add(value["id"])

    return list(entity_ids)


def resolve_wikidata_entity_value(entity_id, label_cache):
    return {
        "id": entity_id,
        "label": label_cache.get(entity_id, {}).get("label", entity_id)
    }


def extract_datavalue(datavalue, label_cache):
    value = datavalue.get("value")

    if isinstance(value, dict):
        if "id" in value:
            return resolve_wikidata_entity_value(value["id"], label_cache)

        if "time" in value:
            return {
                "time": value.get("time"),
                "precision": value.get("precision"),
                "calendar_model": value.get("calendarmodel")
            }

        if "latitude" in value and "longitude" in value:
            return {
                "latitude": value.get("latitude"),
                "longitude": value.get("longitude"),
                "precision": value.get("precision"),
                "globe": value.get("globe")
            }

        if "amount" in value:
            return {
                "amount": value.get("amount"),
                "unit": value.get("unit")
            }

        return value

    return value


def extract_claim_values(entity, property_id, label_cache):
    values = []

    for claim in entity.claims.get(property_id, []):
        mainsnak = claim.get("mainsnak", {})
        datavalue = mainsnak.get("datavalue")

        if not datavalue:
            continue

        values.append(extract_datavalue(datavalue, label_cache))

    return values


def simplify_entity(entity, language="en"):
    property_ids = list(entity.claims.keys())

    property_cache = update_property_cache(property_ids, language)

    referenced_entity_ids = collect_referenced_entity_ids(entity)
    label_cache = update_label_cache(referenced_entity_ids, language)

    properties = {}

    for property_id in property_ids:
        # external-id properties were skipped and are not in cache
        if property_id not in property_cache:
            continue

        property_info = property_cache[property_id]
        property_name = property_info["label"]

        values = extract_claim_values(
            entity,
            property_id,
            label_cache
        )

        properties[property_name] = {
            "property_id": property_id,
            "datatype": property_info["datatype"],
            "values": values
        }

    return {
        "id": entity.id,
        "label": entity.labels.get(language, {}).get("value"),
        "description": entity.descriptions.get(language, {}).get("value"),
        "properties": properties
    }


def get_all_properties(entity, language="en"):
    property_ids = list(entity.claims.keys())
    property_cache = update_property_cache(property_ids, language)

    return [
        {
            "property_id": property_id,
            "property_name": property_cache[property_id]["label"],
            "datatype": property_cache[property_id]["datatype"]
        }
        for property_id in property_ids
        if property_id in property_cache
    ]


def save_json_to_file(data, output_file):
    if isinstance(data, BaseModel):
        data = data.model_dump()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    #entity_id = "Q151005" # Battle of Leipzig
    #entity_id = "Q179591"   # Battle of Cannae
    #entity_id = "Q83224"    # Battle of Hastings      
    #entity_id = "Q7222682"    # Second Turkish Siege of Vienna
    entity_id = "Q200056"    # Battle of Zama

    print(f"Retrieving entity {entity_id}...")
    page = retrieve_wikidata_page_by_id(entity_id)
    entity = get_entity(page, entity_id)
    properties = get_all_properties(entity)

    print("Simplifying entity...")
    simplified = simplify_entity(entity)

    print("Saving JSON to files...")
    save_json_to_file(entity, f"extracted/{entity_id}_basic.json")
    save_json_to_file(simplified, f"extracted/{entity_id}_simplified.json")
    save_json_to_file(properties, f"extracted/{entity_id}_properties.json")

    properties = get_all_properties(entity)
    print(f"Found {len(properties)} statement properties:")
    for prop in properties:
        print(prop)