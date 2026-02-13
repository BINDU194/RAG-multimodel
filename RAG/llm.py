def ask_llm(context, question, api_key, model):
    """Call Groq LLM to generate an answer.

    This function imports the Groq client lazily so the module can be
    imported even if the `groq` package is not installed. When the
    package is missing, we raise a clear error message that the caller
    (the app) can surface to the user instead of failing at import time.
    """
    try:
        from groq import Groq
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'groq'. Install it with: pip install groq-python-client"
        ) from e

    client = Groq(api_key=api_key)

    prompt = f"""
Answer only using the context below.
If the answer is not present, respond with: "Not enough information in the provided context."

Context:
{context}

Question: {question}

Answer:
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()
