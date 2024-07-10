# Claudia: An AI-powered Coding Assistant

Claudia is an AI-powered coding assistant built on top of the Anthropic API. It's designed to help developers with various coding tasks, provide explanations, and even execute Python code in a controlled environment.

## Features

- Engage in natural language conversations about coding topics
- Execute Python code safely within the chat interface
- Provide explanations and breakdowns of complex coding concepts
- Utilize a variety of tools to assist with coding tasks
- Stream responses for real-time interaction

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/wangjing0/Claudia.git
   cd Claudia/claudia
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Anthropic API key:
   - Create a `.env` file in the project root
   - Add your Anthropic API key: `ANTHROPIC_API_KEY=your_api_key_here`

## Usage

### Basic Usage

To start a conversation with Claudia:

```python
from claudia import Claudia

chat = Claudia()
response = chat("How do I implement a binary search in Python?")
print(response)
```

### CoderClaudia

To use Claudia with code execution capabilities:

```python
from claudia import CoderClaudia

CLAUDE_MODEL = 'claude-3-5-sonnet-20240620'
coder = CoderClaudia(model=CLAUDE_MODEL, ask=True)

task = '''find the smallest integer that has its square root larger the meaning of life. hint: use binary search''' 

_ = coder.run(prompt=task, max_loops=3)
```

### Customization

You can customize Claudia's behavior by providing a system prompt:

```python
system_prompt = """
You are a Python expert. Always provide code examples in your explanations.
Focus on best practices and efficient solutions.
"""

coder = CoderClaudia(model=CLAUDE_MODEL, system_prompt=system_prompt)
```

## Advanced Features

### Streaming Responses

To stream responses in real-time:

```python
for chunk in chat("Explain object-oriented programming", stream=True):
    print(chunk, end='', flush=True)
```

### Using Custom Tools

You can extend Claudia's capabilities by adding custom tools:

```python
def custom_tool(arg1, arg2):
    # Tool implementation
    return result

chat = Claudia(tools=[custom_tool])
```

## Contributing

Contributions to Claudia are welcome! Please refer to the `CONTRIBUTING.md` file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Claudia is powered by the Anthropic API
- Special thanks to all contributors and users of Claudia

For more information and updates, please visit [https://github.com/wangjing0/Claudia](https://github.com/wangjing0/Claudia).