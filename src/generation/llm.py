"""
Generation de reponses avec Groq API (cloud).
"""
import os
from groq import Groq

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
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

    def generate(self, prompt, temperature=0.3, max_tokens=1500, reasoning_effort="low"):
        if not hasattr(self, 'client'):
            return "[ERREUR] Client Groq non initialisé (clé API manquante)."

        try:
            # Appel natif à l'API Groq (gère automatiquement la bonne URL et les en-têtes)
            # NOTE : openai/gpt-oss-* sont des modèles de raisonnement. Le raisonnement caché
            # consomme des tokens comptés dans max_completion_tokens -> on augmente le budget
            # et on force un effort de raisonnement bas pour laisser de la place à la réponse.
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Tu es un assistant scientifique specialise en climatologie."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            if finish_reason == "length":
                # La réponse a été coupée faute de tokens -> on le signale au lieu de
                # renvoyer silencieusement une demi-phrase.
                print("⚠️ Réponse tronquée (finish_reason='length'). Augmente max_tokens si ça se reproduit.")

            return (content or "").strip()
        except Exception as e:
            return f"[ERREUR] {str(e)}"

    def is_available(self):
        return self.api_key is not None and len(self.api_key) > 10