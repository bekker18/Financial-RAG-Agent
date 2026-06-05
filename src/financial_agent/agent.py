import json
from typing import Any

from financial_agent.llm import GroqLLM
from financial_agent.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

SYSTEM_PROMPT = """
You are a financial research assistant.

Rules:
1. Use only information from retrieved SEC filing chunks.
2. Cite every important claim using:
   - ticker
   - filing form
   - filing date
   - source_id
   - SEC URL when available
3. Do not call the CIK a source ID.
4. If the retrieved documents are old, say that the answer is based only on indexed filings.
5. Do not invent financial numbers.
6. Do not provide investment advice or buy/sell recommendations.
"""


class FinancialResearchAgent:
    def __init__(self) -> None:
        self.llm = GroqLLM()

    def answer(self, question: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        # We allow a few tool-calling rounds.
        for _ in range(4):
            response = self.llm.chat(
                messages=messages,
                tools=TOOL_SCHEMAS,
                temperature=0.1,
            )

            assistant_message = response.choices[0].message

            tool_calls = assistant_message.tool_calls

            if not tool_calls:
                return assistant_message.content or ""

            assistant_dict: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in tool_calls
                ],
            }

            messages.append(assistant_dict)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                raw_arguments = tool_call.function.arguments

                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    arguments = {}

                function = TOOL_FUNCTIONS.get(function_name)

                if function is None:
                    tool_result = json.dumps(
                        {"error": f"Unknown tool: {function_name}"}
                    )
                else:
                    try:
                        tool_result = function(**arguments)
                    except Exception as exc:
                        tool_result = json.dumps(
                            {
                                "error": str(exc),
                                "tool": function_name,
                            }
                        )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": tool_result,
                    }
                )

        return (
            "I could not complete the answer within the tool-call limit. "
            "Try asking a narrower question."
        )
