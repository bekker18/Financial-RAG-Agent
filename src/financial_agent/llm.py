from groq import Groq

from financial_agent.config import get_settings


class GroqLLM:
    def __init__(self) -> None:
        self.settings = get_settings()

        if not self.settings.groq_api_key:
            raise ValueError("Missing GROQ_API_KEY in environment.")

        self.client = Groq(api_key=self.settings.groq_api_key)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.1,
    ):
        kwargs = {
            "model": self.settings.groq_model,
            "messages": messages,
            "temperature": temperature,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        return self.client.chat.completions.create(**kwargs)
