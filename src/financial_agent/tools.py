import json

from financial_agent.vector_store import FinancialVectorStore

"""
We give the model tool definitions,
the model returns structured tool calls, our code executes them,
then we return the tool result back to the model.

TOOL_SCHEMAS
    Describes tools to LLM.

TOOL_FUNCTIONS
    Maps tool names to actual Python functions.
"""


def search_financial_documents(
    query: str,
    ticker: str | None = None,
    top_k: int = 5,
) -> str:
    """
    RAG search tool.

    Returns json string because LLM tool results should be serializable.
    """

    store = FinancialVectorStore()
    chunks = store.search(query=query, ticker=ticker, top_k=top_k)

    results = []

    for chunk in chunks:
        meta = chunk.metadata

        results.append(
            {
                "source_id": chunk.id,
                "ticker": meta.get("ticker"),
                "company_name": meta.get("company_name"),
                "form": meta.get("form"),
                "filing_date": meta.get("filing_date"),
                "url": meta.get("url"),
                "distance": chunk.distance,
                "text": chunk.text[:1800],
            }
        )

    return json.dumps(results, indent=2)


def calculate_growth_rate(
    old_value: float,
    new_value: float,
) -> str:
    """
    Calculate percentage growth from old_value to new_value.
    """

    if old_value == 0:
        return json.dumps({"error": "old_value can't be zero."})

    growth = ((new_value - old_value) / old_value) * 100

    return json.dumps(
        {
            "old_value": old_value,
            "new_value": new_value,
            "growth_percent": growth,
        },
        indent=2,
    )


def calculate_ratio(
    numerator: float,
    denominator: float,
) -> str:
    """
    Generic financial ratio calculator.
    """

    if denominator == 0:
        return json.dumps({"error": "denominator can't be zero."})

    return json.dumps(
        {
            "numerator": numerator,
            "denominator": denominator,
            "ratio": numerator / denominator,
        },
        indent=2,
    )


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_financial_documents",
            "description": (
                "Search indexed SEC filings for relevant information. "
                "Use this whenever the user asks about a company's risks, "
                "business, revenue, margins, debt, strategy, filings, "
                "or financial performance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for the financial documents.",
                    },
                    "ticker": {
                        "type": "string",
                        "description": "Optional stock ticker, for example AAPL or MSFT.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_growth_rate",
            "description": "Calculate percentage growth from old value to new value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_value": {"type": "number"},
                    "new_value": {"type": "number"},
                },
                "required": ["old_value", "new_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_ratio",
            "description": "Calculate a ratio using numerator / denominator.",
            "parameters": {
                "type": "object",
                "properties": {
                    "numerator": {"type": "number"},
                    "denominator": {"type": "number"},
                },
                "required": ["numerator", "denominator"],
            },
        },
    },
]


TOOL_FUNCTIONS = {
    "search_financial_documents": search_financial_documents,
    "calculate_growth_rate": calculate_growth_rate,
    "calculate_ratio": calculate_ratio,
}
