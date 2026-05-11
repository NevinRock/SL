from openai import OpenAI

client = OpenAI(
    api_key="sk-8dc0ec1847bc4f068af9e5ca517e5bae",
    base_url="https://api.deepseek.com",
)


def gloss_to_sentence(gloss_sentence: str) -> str:

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content":
                        "You are a robot instruction translator. "
                        "Convert sign language gloss into short imperative English commands "
                        "for a vision-language-action robot. "
                        "Do not use first person pronouns. "
                        "Do not use past tense. "
                        "Output only the command sentence."
                },
                {
                    "role": "user",
                    "content": gloss_sentence
                }
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("DeepSeek Error:", e)
        return gloss_sentence