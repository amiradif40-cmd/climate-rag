"""
Generation de reponses avec Groq API (cloud).
"""
import os
from groq import Groq

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
class GroqLLM:
    """
    Client LLM via API Groq - ultra rapide, zero RAM locale.
    Parfait pour un projet demo/testable publiquement.
    """

    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or LLM_MODEL

        if not self.api_key:
            print("Cle API Groq manquante.")
            print("Cree-en une gratuite sur https://console.groq.com")
        else:
            # Initialisation du client officiel Groq
            self.client = Groq(api_key=self.api_key)

    def generate(self, prompt, temperature=0.3, max_tokens=600):
        if not hasattr(self, 'client'):
            return "[ERREUR] Client Groq non initialisé (clé API manquante)."

        try:
            # Appel natif à l'API Groq (gère automatiquement la bonne URL et les en-têtes)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Tu es un assistant scientifique specialise en climatologie."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[ERREUR] {str(e)}"

    def is_available(self):
        return self.api_key is not None and len(self.api_key) > 10