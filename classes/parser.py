import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage # Import these classes

class Parser: # Assuming your Parser class
    @staticmethod # Or make it a regular method if it uses self
    def parse_chunk_to_human_readable(chunk):
        """
        Parses a given chunk (dictionary) from the agent's stream into a human-readable format.

        Args:
            chunk (dict): A dictionary representing a chunk from the agent's stream.

        Returns:
            str: A human-readable string representation of the chunk.
        """
        human_readable_output = []

        if 'agent' in chunk:
            for message in chunk['agent']['messages']:
                if message.type == 'ai':
                    if message.content:
                        human_readable_output.append(f"Agent (Thought/Content):\n{message.content}")
                    if message.tool_calls:
                        human_readable_output.append("Agent (Tool Calls):")
                        for tool_call in message.tool_calls:
                            human_readable_output.append(f"  Tool Name: {tool_call['name']}")
                            human_readable_output.append(f"  Arguments: {tool_call['args']}")
                elif message.type == 'human':
                    human_readable_output.append(f"User: {message.content}")

        if 'tools' in chunk:
            for message in chunk['tools']['messages']:
                if message.type == 'tool':
                    human_readable_output.append(f"Tool Output ({message.name}):")
                    try:
                        # Attempt to parse the content as JSON if it's a string, as per the tool's return format
                        tool_content = json.loads(message.content)
                        if isinstance(tool_content, dict) and 'output' in tool_content:
                            human_readable_output.append(f"  Output: {tool_content['output']}")
                        elif isinstance(tool_content, dict) and 'error' in tool_content:
                            human_readable_output.append(f"  Error: {tool_content['error']}")
                        else:
                            human_readable_output.append(f"  Raw Content: {tool_content}")
                    except json.JSONDecodeError:
                        human_readable_output.append(f"  Raw Content: {message.content}")

        return "\n\n".join(human_readable_output)