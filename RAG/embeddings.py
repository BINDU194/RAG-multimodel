import requests
import numpy as np


def get_jina_embeddings(texts, api_key):
    url = "https://api.jina.ai/v1/embeddings"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "jina-embeddings-v4",
        "input": texts
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        # bubble up with clearer context for the caller
        status = getattr(e.response, 'status_code', None)
        msg = f"Jina embeddings request failed with status {status}: {e.response.text if e.response is not None else ''}"
        raise requests.exceptions.HTTPError(msg) from e
    except requests.exceptions.RequestException as e:
        raise

    vectors = [item["embedding"] for item in response.json()["data"]]
    return np.array(vectors).astype("float32")
