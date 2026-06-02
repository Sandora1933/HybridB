import json
import os
import requests
from pydantic import BaseModel, Field, ConfigDict


PROPERTY_CACHE_FILE = "wikidata_property_cache.json"

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


def load_property_cache():
    if os.path.exists(PROPERTY_CACHE_FILE):
        with open(PROPERTY_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    return {}


def save_property_cache(cache):
    with open(PROPERTY_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def retrieve_entities(ids):
    ids_string = "|".join(ids)

    url = (
        "https://www.wikidata.org/w/api.php"
        "?action=wbgetentities"
        "&format=json"
        "&languages=en"
        "&ids=" + ids_string
    )

    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    return response.json()["entities"]


def update_property_cache(property_ids):
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
                           .get("en", {})
                           .get("value", property_id),
            "datatype": datatype
        }

    save_property_cache(cache)

    return cache


def retrieve_wikidata_page_by_id(entity_id):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"

    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    return WikidataPage.model_validate(response.json())


def get_entity(page, entity_id):
    return page.entities[entity_id]


def get_entity_label(entity_id, language="en"):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"

    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    data = response.json()
    entity_data = data["entities"][entity_id]

    return entity_data.get("labels", {}).get(language, {}).get("value", entity_id)


def resolve_wikidata_entity_value(entity_id, language="en"):
    return {
        "id": entity_id,
        "label": get_entity_label(entity_id, language)
    }


def extract_datavalue(datavalue, language="en"):
    value = datavalue.get("value")

    if isinstance(value, dict):
        if "id" in value:
            return resolve_wikidata_entity_value(value["id"], language)

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


def extract_claim_values(entity, property_id, language="en"):
    values = []

    for claim in entity.claims.get(property_id, []):
        mainsnak = claim.get("mainsnak", {})
        datavalue = mainsnak.get("datavalue")

        if not datavalue:
            continue

        values.append(extract_datavalue(datavalue, language))

    return values


def simplify_entity(entity, language="en"):
    property_ids = list(entity.claims.keys())
    property_cache = update_property_cache(property_ids)

    properties = {}

    for property_id in property_ids:

        # Property was skipped because it is an external-id
        if property_id not in property_cache:
            continue

        property_info = property_cache[property_id]

        property_name = property_info["label"]

        values = extract_claim_values(
            entity,
            property_id,
            language
        )

        properties[property_name] = {
            "property_id": property_id,
            "values": values
        }

    return {
        "id": entity.id,
        "label": entity.labels.get(language, {}).get("value"),
        "description": entity.descriptions.get(language, {}).get("value"),
        "properties": properties
    }


def get_all_properties(entity):
    property_ids = list(entity.claims.keys())
    property_cache = update_property_cache(property_ids)

    return [
        {
            "property_id": property_id,
            "property_name": property_cache[property_id]["label"]
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
    entity_id = "Q151005"

    print(f"Retrieving entity {entity_id}...")
    page = retrieve_wikidata_page_by_id(entity_id)
    entity = get_entity(page, entity_id)

    print("Simplifying entity...")
    simplified = simplify_entity(entity)

    print("Saving JSON to files...")
    save_json_to_file(entity, "Q151005_basic.json")
    save_json_to_file(simplified, "Q151005_simplified.json")

    properties = get_all_properties(entity)

    save_json_to_file(properties, "Q151005_properties.json")

    print(f"Found {len(properties)} statement properties:")
    for prop in properties:
        print(prop)