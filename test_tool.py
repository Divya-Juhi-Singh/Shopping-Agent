from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool

load_dotenv()

@tool
def get_rating(product_id: int) -> str:
    """Get a product rating using its integer product ID."""
    return f"Rating for product {product_id} is 4.5"

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)

response = llm.bind_tools([get_rating]).invoke(
    "Get the rating for product ID 1."
)

print(response.tool_calls)