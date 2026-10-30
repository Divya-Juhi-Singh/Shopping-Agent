
# eval_shopping_agent.py

import json
import uuid

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from shopping_agent import agent, search_products, get_ratings

load_dotenv()

# Use a separate model call to judge response quality.
judge_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)

# ---------------------------------------------------------
# 1. TOOL-CALL ACCURACY DATASET
# ---------------------------------------------------------

TOOL_TESTS = [
    {
        "query": "Find organic honey under $20.",
        "expected_tool": "search_products",
        "expected_args": {
            "is_organic": True,
            "max_price": 20,
            "query_contains": "honey",
        },
    },
    {
        "query": "Find non-organic almonds below $10.",
        "expected_tool": "search_products",
        "expected_args": {
            "is_organic": False,
            "max_price": 10,
            "query_contains": "almond",
        },
    },
]


def invoke_agent(query: str):
    """Invoke the real agent with a fresh thread for an isolated test."""
    config = {
        "configurable": {
            "thread_id": f"eval-{uuid.uuid4()}"
        }
    }

    return agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
    )


def collect_tool_calls(result):
    """Extract tool calls from all AI messages in the agent result."""
    calls = []

    for message in result["messages"]:
        if getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                calls.append({
                    "name": call["name"],
                    "args": call.get("args", {}),
                })

    return calls


def check_tool_test(test):
    print(f"\nQuery: {test['query']}")

    try:
        result = invoke_agent(test["query"])
        calls = collect_tool_calls(result)
    except Exception as exc:
        print(f"FAIL — agent invocation error: {exc}")
        return False

    expected_name = test["expected_tool"]
    expected = test["expected_args"]

    matching_calls = [
        call for call in calls
        if call["name"] == expected_name
    ]

    if not matching_calls:
        print(f"FAIL — expected tool {expected_name!r} was not called.")
        print("Actual tool calls:", calls)
        return False

    # Check whether at least one call has the expected arguments.
    for call in matching_calls:
        args = call["args"]

        if (
            args.get("is_organic") == expected["is_organic"]
            and args.get("max_price") == expected["max_price"]
            and expected["query_contains"].lower()
            in str(args.get("query", "")).lower()
        ):
            print("PASS — expected tool and arguments found.")
            print("Matched call:", call)
            return True

    print("FAIL — tool was called, but arguments did not match.")
    print("Expected:", expected)
    print("Actual calls:", matching_calls)
    return False


def run_tool_evaluations():
    print("\n========== TOOL-CALL ACCURACY ==========")

    results = [check_tool_test(test) for test in TOOL_TESTS]
    passed = sum(results)

    print(f"\nTool-call accuracy: {passed}/{len(results)} tests passed")
    return passed, len(results)


# ---------------------------------------------------------
# 2. RESPONSE QUALITY: LLM-AS-A-JUDGE
# ---------------------------------------------------------

def get_reference_products():
    """
    Build reference facts from the real product database and ratings API.
    This avoids asking the judge to grade against invented product data.
    """
    raw_products = search_products.invoke({
        "query": "honey",
        "max_price": 20,
        "is_organic": True,
    })

    products = json.loads(raw_products)
    references = []

    for product in products:
        rating_data = json.loads(
            get_ratings.invoke({"product_id": product["id"]})
        )

        average_rating = rating_data.get("average_rating")

        if average_rating is not None and average_rating >= 4:
            references.append({
                "id": product["id"],
                "name": product["name"],
                "price": product["price"],
                "is_organic": product["is_organic"],
                "average_rating": average_rating,
            })

    return references


def judge_response(query, response, reference_products):
    """Ask an LLM judge to score the response on three dimensions."""

    prompt = f"""
You are a strict evaluator for a shopping assistant.

Evaluate the assistant's response to the user's shopping request.
Use the reference product data as the source of truth.

USER QUERY:
{query}

REFERENCE PRODUCTS:
{json.dumps(reference_products, indent=2)}

ASSISTANT RESPONSE:
{response}

Score each dimension from 1 to 5:

1. relevance:
   Does the response answer the user's shopping request?

2. correctness:
   Are the product names, prices, ratings, and organic claims consistent
   with the reference data? Penalize invented products and unsupported
   claims. If no products qualify, the response should say so honestly.

3. format_compliance:
   If products are recommended, does the response use a numbered list
   with product name, ID, price, rating, and organic status?
   If no products qualify, a clear explanation is acceptable.

Return ONLY valid JSON with this structure:
{{
  "relevance": 1,
  "correctness": 1,
  "format_compliance": 1,
  "reason": "Brief explanation"
}}
"""

    result = judge_llm.invoke(prompt).content.strip()

    # Handle models that wrap JSON in Markdown fences.
    if result.startswith("```"):
        result = result.split("\n", 1)[1]
        result = result.rsplit("```", 1)[0].strip()

    scores = json.loads(result)

    for key in ("relevance", "correctness", "format_compliance"):
        if not isinstance(scores.get(key), (int, float)):
            raise ValueError(f"Invalid judge score for {key}: {scores}")
        if not 1 <= scores[key] <= 5:
            raise ValueError(f"Score out of range for {key}: {scores}")

    return scores


def run_response_quality_evaluation():
    print("\n========== RESPONSE QUALITY ==========")

    query = "Find organic honey under $20 with a rating of at least 4."

    try:
        references = get_reference_products()
        result = invoke_agent(query)
        response = result["messages"][-1].content

        print("\nAgent response:\n", response)

        scores = judge_response(query, response, references)

        print("\nJudge scores:")
        for dimension in ("relevance", "correctness", "format_compliance"):
            print(f"  {dimension}: {scores[dimension]}/5")

        print("Judge explanation:", scores["reason"])

        average = sum(
            scores[key]
            for key in ("relevance", "correctness", "format_compliance")
        ) / 3

        print(f"Average quality score: {average:.2f}/5")

        # Set a baseline threshold for your project.
        passed = all(
            scores[key] >= 4
            for key in ("relevance", "correctness", "format_compliance")
        )

        print("QUALITY EVAL:", "PASS" if passed else "FAIL")
        return passed

    except Exception as exc:
        print(f"Response-quality evaluation failed: {exc}")
        return False


if __name__ == "__main__":
    tool_passed, tool_total = run_tool_evaluations()
    quality_passed = run_response_quality_evaluation()

    print("\n========== FINAL SUMMARY ==========")
    print(f"Tool tests: {tool_passed}/{tool_total} passed")
    print(f"Response quality: {'PASS' if quality_passed else 'FAIL'}")
