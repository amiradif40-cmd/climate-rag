"""
Generation de reponses avec Groq API (cloud).
"""
import os
import requests


LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


class GroqLLM:
    """
    Client LLM via API Groq - ultra rapide, zero RAM locale.
    Parfait pour un projet demo/testable publiquement.
    """

    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or LLM_MODEL
        self.url = "https://api.groq.com/openai/v1/chat/completions"

        if not self.api_key:
            print("Cle API Groq manquante.")
            print("Cree-en une gratuite sur https://console.groq.com")

    def generate(self, prompt, temperature=0.3, max_tokens=600):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Tu es un assistant scientifique specialise en climatologie."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(self.url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[ERREUR] {str(e)}"

    def is_available(self):
        return self.api_key is not None and len(self.api_key) > 10