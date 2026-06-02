import json
import os
import requests
from pydantic import BaseModel, Field, ConfigDict


PROPERTY_CACHE_FILE = "wikidata_property_cache.json"


def load_property_cache():
    if os.path.exists(PROPERTY_CACHE_FILE):
        with open(PROPERTY_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    return {}


def save_property_cache(cache):
    with open(PROPERTY_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            cache,
            f,
            indent=2,
            ensure_ascii=False
        )


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


def retrieve_wikidata_page_by_id(entity_id):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"

    response = requests.get(
        url,
        headers={
            "User-Agent": "WikidataBattleRetriever/1.0 (vlad8aromanov@gmail.com)"
        },
        timeout=30
    )
    response.raise_for_status()

    return WikidataPage.model_validate(response.json())


def get_entity(page, entity_id):
    return page.entities[entity_id]


def get_entity_label(entity_id, language="en"):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"

    response = requests.get(
        url,
        headers={
            "User-Agent": "WikidataBattleRetriever/1.0 (vlad8aromanov@gmail.com)"
        },
        timeout=30
    )
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
        # Q-item or property reference
        if "id" in value:
            return resolve_wikidata_entity_value(value["id"], language)

        # Time value
        if "time" in value:
            return {
                "time": value.get("time"),
                "precision": value.get("precision"),
                "calendar_model": value.get("calendarmodel")
            }

        # Coordinates
        if "latitude" in value and "longitude" in value:
            return {
                "latitude": value.get("latitude"),
                "longitude": value.get("longitude"),
                "precision": value.get("precision"),
                "globe": value.get("globe")
            }

        # Quantity
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


def resolve_property_name(property_id, language="en"):
    return get_entity_label(property_id, language)


def simplify_entity(entity, language="en"):
    properties = {}

    for property_id in entity.claims.keys():
        property_name = resolve_property_name(property_id, language)
        values = extract_claim_values(entity, property_id, language)

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


def save_json_to_file(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_all_properties(entity, language="en"):
    properties = []

    for property_id in entity.claims.keys():
        properties.append({
            "property_id": property_id,
            "property_name": resolve_property_name(property_id, language)
        })

    return properties


if __name__ == "__main__":
    entity_id = "Q151005"

    print(f"Retrieving entity {entity_id}...")
    page = retrieve_wikidata_page_by_id(entity_id)
    entity = get_entity(page, entity_id)

    print("Simplifying entity...")
    simplified = simplify_entity(entity)

    print("Saving JSON to a file...")
    save_json_to_file(entity.model_dump(), "Q151005_basic.json")
    save_json_to_file(simplified, "Q151005_simplified.json")

    properties = get_all_properties(entity)
    print(f"Found {len(properties)} properties:")
    for prop in properties:
        print(prop)