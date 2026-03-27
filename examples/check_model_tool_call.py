from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def calculator(a: float, b: float, operation: str) -> float:
    """Perform basic arithmetic. operation: 'add','subtract','multiply','divide'."""
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            raise ValueError("Division by zero")
        return a / b
    raise ValueError(f"Invalid operation: {operation}")


llm = ChatOllama(model="mistral-small3.2:24b", temperature=0.0)

SYSTEM_PROMPT = """You are a math assistant.
For every arithmetic question, call the calculator tool.
After receiving the result, respond with only the final numeric answer."""

agent = create_agent(
    model=llm,
    tools=[calculator],
    system_prompt=SYSTEM_PROMPT,
)

# Step 1: verify the model emits structured tool calls
llm_with_tools = llm.bind_tools([calculator])
resp = llm_with_tools.invoke("What is 12456 divided by 7.143?")
print("Tool calls:", resp.tool_calls)  # Must be non-empty to proceed

# Step 2: test the agent with the correct input format
if resp.tool_calls:
    agent_response = agent.invoke({"messages": [{"role": "user", "content": "What is 12456 divided by 7.143?"}]})
    # Print just the last message content (the final answer)
    print("Agent response:", agent_response["messages"][-1].content)
else:
    print("Model did not emit tool calls — agent won't work with this model.")
