import wikipediaapi


def save_text_to_file(text, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)


def retrieve_wikipedia_page(
    page_title,
    language="en"
):
    wiki = wikipediaapi.Wikipedia(
        user_agent="BattleRetriever/1.0 (vlad8aromanov@gmail.com)",
        language=language
    )

    page = wiki.page(page_title)

    if not page.exists():
        raise ValueError(
            f"Wikipedia page '{page_title}' does not exist."
        )

    return {
        "title": page.title,
        "summary": page.summary,
        "text": page.text,
        "url": page.fullurl,
        "categories": list(page.categories.keys())
    }


if __name__ == "__main__":
    page_title = "Battle_of_Vienna"
    language = "en"
    file_to_save = "extracted_wikipedia/battle_of_vienna.txt"

    try:
        page_data = retrieve_wikipedia_page(page_title, language)
        print(f"Title: {page_data['title']}")
        print(f"Summary: {page_data['summary']}")
        print(f"URL: {page_data['url']}")
        print(f"Categories: {', '.join(page_data['categories'])}")
        save_text_to_file(page_data["text"], file_to_save)
        print(f"Saved to '{file_to_save}'")
    except ValueError as e:
        print(e)