
"""Detect emotions in text using the IBM Watson NLP service."""

import requests


URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}


def emotion_detector(text_to_analyze):
    """Send text to the Watson NLP emotion model and return its JSON response."""
    payload = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(URL, headers=HEADERS, json=payload, timeout=30)
    response.raise_for_status()
    return response.text
