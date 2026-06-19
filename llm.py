import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"
SCADSAI_API_KEY = os.getenv("SCADSAI_API_KEY")
SCADSAI_BASE_URL = "https://llm.scads.ai/v1"


class ScadsChatModel:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name

        if SCADSAI_API_KEY is None:
            raise ValueError(
                "SCADSAI API key not found. "
                "Please set SCADSAI_API_KEY in your .env file."
            )

        self.client = OpenAI(
            base_url=SCADSAI_BASE_URL,
            api_key=SCADSAI_API_KEY
        )

    def generate_response(
        self,
        user_message,
        system_message=None,
        max_new_tokens=512,
        do_sample=False,
        temperature=None,
        top_p=None
    ):
        messages = []

        if system_message:
            messages.append({
                "role": "system",
                "content": system_message
            })

        messages.append({
            "role": "user",
            "content": user_message
        })

        generation_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens
        }

        if do_sample:
            generation_kwargs["temperature"] = (
                temperature if temperature is not None else 0.7
            )
            generation_kwargs["top_p"] = (
                top_p if top_p is not None else 0.9
            )
        else:
            generation_kwargs["temperature"] = 0.0

        response = self.client.chat.completions.create(
            **generation_kwargs
        )

        choice = response.choices[0]
        message = choice.message

        content = message.content

        if content is None:
            print("Raw response:")
            print(response.model_dump_json(indent=2))

            raise ValueError(
                f"Model returned no content. finish_reason={choice.finish_reason}"
            )

        return content.strip()


def load_llm(model_name=MODEL_NAME):
    return ScadsChatModel(model_name=model_name)


def generate_answer(
    llm,
    prompt,
    system_message=None,
    max_new_tokens=1024,
    do_sample=False,
    temperature=None,
    top_p=None
):
    return llm.generate_response(
        user_message=prompt,
        system_message=system_message,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p
    )