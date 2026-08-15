"""
Génération de réponses avec Groq API (cloud).
Idéal pour déploiement public (LinkedIn, portfolio).
"""
import os
import requests


LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


class GroqLLM:
    """
    Client LLM via API Groq - ultra rapide, zéro RAM locale.
    Parfait pour un projet démo/testable publiquement.
    """

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or LLM_MODEL
        self.url = "https://api.groq.com/openai/v1/chat/completions"

        if not self.api_key:
            print("⚠️  Clé API Groq manquante.")
            print("   Crée-en une gratuite sur https://console.groq.com")

    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1500) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Tu es un assistant scientifique spécialisé en climatologie."},
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

    def is_available(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 10