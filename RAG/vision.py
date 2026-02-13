import base64


def describe_image(image_bytes, api_key):
    """Describe an image using Groq Vision.

    The Groq client is imported lazily so the module can be imported even
    when the optional `groq` package is not installed. If the package is
    missing, a ModuleNotFoundError with a helpful message is raised.
    """
    try:
        from groq import Groq
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'groq'. Install it with: pip install groq-python-client"
        ) from e

    client = Groq(api_key=api_key)

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image clearly."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    }
                ]
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()
