import json
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
#from classes.vector_store import VectorStore
#from classes.classifier import Classifier
from classes.file_manager import FileManager
from classes.sandbox import Sandbox
from classes.parser import Parser


from classes.sandbox import Sandbox

sandbox = Sandbox()

@tool
def execute_code(code: str) -> str:
    """
    Executes the given code in a separate process and returns output or errors. Note that a "print" statement is needed to return any output.

    Args:
        code (str): Python code to execute.

    Returns:
        dict: {'output': str} or {'error': str}
    """
    sandbox_result = sandbox.execute(code)
    # Ensure the result is always a dictionary with 'output' or 'error'
    if "output" not in sandbox_result and "error" not in sandbox_result:
        sandbox_result = {"output": "", "error": "Unexpected sandbox output format."}
    # LangChain's ToolMessage content expects a JSON string
    #print(f"DEBUG: execute_code tool returning: {sandbox_result}") # Add this line
    return json.dumps(sandbox_result)
    

if __name__ == "__main__":
    llm = ChatOllama(model="qwen3:8b", temperature=0.5)

    tools = [execute_code]

    agent = create_react_agent(
        model=llm,
        tools=tools,
    )
    # Run a prompt

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "Define and execute a python script that exceeds your ability as an LLM. Use standard libraries and make sure it runs in under 60 seconds. Return the code and the result"}]},
        stream_mode="updates"
    ):
        print(Parser.parse_chunk_to_human_readable(chunk))
        print("\n")
