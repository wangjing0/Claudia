import os, inspect, typing, base64, json
from collections import abc
from typing import List, Dict, Any, Union, Tuple, Optional
from anthropic import Anthropic
from anthropic.types import Usage, TextBlock, Message, ToolUseBlock
from anthropic.resources import messages
from toolslm.funccall import first, get_schema, nested_idx, noop
from toolslm.shell import get_shell
from fastcore.meta import delegates
from fastcore.utils import patch
from dotenv import load_dotenv, find_dotenv
from claudia_utils import print_colored
from claudia_utils import USER_COLOR, CLAUDE_COLOR, TOOL_COLOR, RESULT_COLOR, ERROR_COLOR, REMINDER_COLOR, ASK_COLOR
import traceback
from IPython.display import display, Markdown

load_dotenv(find_dotenv())

# Monkey patching for anthropic types
@patch
def __repr__(self:Usage): 
    """String representation of Usage."""
    return f'Token Usage: In: {self.input_tokens}; Out: {self.output_tokens}.'
@patch
def __add__(self:Usage, b):
    "Add together each of `input_tokens` and `output_tokens`"
    return usage(self.input_tokens + b.input_tokens, self.output_tokens + b.output_tokens)


def find_block(r:abc.Mapping, # The message to look in
               block_type:type=TextBlock  # The type of block to find
              ):
    "Find the first block of type `block_type` in `r.content`."
    return first(o for o in r.content if isinstance(o, block_type))

def contents(r):
    """Helper to get the contents from Claude response `r`.
        
    Args:
        r (Message): The message to extract contents from
    
    Returns:
        str: The extracted content
    """
    block = find_block(r)
    if not block and r.content: 
        block = r.content[0]
    return block.text.strip() if hasattr(block,'text') else str(block)


def usage(input=0, output=0):
    """
    Create a Usage object with the given input and output token counts.
    
    Args:
        input (int): Number of input tokens
        output (int): Number of output tokens
    
    Returns:
        Usage: A Usage object
    """
    return Usage(input_tokens=input, output_tokens=output)


class Client:
    def __init__(self, model: str = 'claude-3-5-sonnet-20240620'):
        "Basic Anthropic messages client."
        self.model = model
        self.use = usage()
        self.client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

    def _record(self, r:Message, prefix=''):
        """
        Record the result `r` and update `use`.
        
        Args:
            r (Message): The message to record
            prefix (str): Optional prefix to add to the message
        
        Returns:
            Message: The recorded message
        """

        if prefix:
            block = find_block(r)
            block.text = prefix + (block.text or '')
        self.result = r
        self.use += r.usage
        return r
    
    def _stream(self, messages:list, prefix='', **kwargs):
        """
        Stream the response from Claude.
        
        Args:
            messages (list): List of messages in the dialog
            prefix (str): Optional prefix to pass to Claude as start of its response
            **kwargs: Additional keyword arguments for the stream
        
        Yields:
            str: Chunks of the streamed response
        """
        with self.client.messages.stream(model=self.model, messages=self._create_messages(messages), **kwargs) as s:
            if prefix: 
                yield(prefix)
            yield from s.text_stream
            self._record(s.get_final_message(), prefix)

    # creating messages
    def _create_message(self, content, role='user', **kwargs) -> dict:
        """
        Helper to create a `dict` appropriate for a Claude message.
        
        Args:
            content (Union[str, dict, Message]): The content of the message
            role (str): The role of the message sender (default: 'user')
            **kwargs: Additional key/value pairs to add to the message
        
        Returns:
            dict: A dictionary representing the message
        """
        if hasattr(content, 'content'): 
            content, role = content.content, content.role
        if isinstance(content, abc.Mapping): 
            content=content['content']
        return dict(role=role, content=content, **kwargs)

    def _create_messages(self, messages:list, **kwargs) -> list[dict]:
        " set 'assistant' role on alternate messages."
        if isinstance(messages,str): 
            messages=[messages]
        return [self._create_message(o, ('user','assistant')[i%2], **kwargs) for i,o in enumerate(messages)]

    def _create_namespace(self, *funcs:list[callable]) -> dict[str,callable]:
        """
        Create a `dict` of name to function in `funcs`, to use as a namespace.
        
        Args:
            *funcs: List of functions to include in the namespace
        Returns:
            dict[str, callable]: A dictionary mapping function names to functions
        """
        return {f.__name__:f for f in funcs}

    def call_func(self, 
                tub:ToolUseBlock, # Tool use block from Claude's message
                namespace: Optional[abc.Mapping]=None, # Namespace to search for tools, defaults to `globals()`
                obj=None, # Class to search for tools
                ) -> dict:
        "Call the function in the tool response `tr`, using namespace `namespace`."
        if namespace is None: 
            namespace = globals()
        if not isinstance(namespace, abc.Mapping): 
            namespace = self._create_namespace(*namespace)
        func = getattr(obj, tub.name, None)
        if not func: 
            func = namespace[tub.name]
        res = func(**tub.input)
        return dict(type="tool_result", tool_use_id=tub.id, content=str(res))   


    def create_toolresponse(self,
            r:abc.Mapping, # Tool use request response from Claude
            namespace:Optional[abc.Mapping]=None, # Namespace to search for tools
            obj=None, # Class to search for tools
        ) -> list[dict]:
        "Create a `tool_result` message from response `r`."
        contents = getattr(r, 'content', [])
        res = [self._create_message(r)]
        toolcontents = [self.call_func(o, namespace=namespace, obj=obj) for o in contents if isinstance(o, ToolUseBlock)]
        if toolcontents: 
            res.append(self._create_message(toolcontents))
        return res
    
    @delegates(messages.Messages.create)
    def __call__(self,
                messages:list, 
                system_prompt='',
                temperature=0, 
                max_tokens=4096, 
                prefix='', 
                stream:bool=False, 
                **kwargs)-> Union[Message, typing.Generator[str, None, None]]:
        """
        Make a call to Claude.
        
        Args:
            messages (list): List of messages in the dialog
            system_prompt (str): The system prompt
            temperature (float): Temperature for response generation
            max_tokens (int): Maximum tokens in the response
            prefix (str): Optional prefix to pass to Claude as start of its response
            stream (bool): Whether to stream the response
            **kwargs: Additional keyword arguments for message creation
        
        Returns:
            Union[Message, typing.Generator[str, None, None]]: The response from Claude
        """
        pref = [prefix.strip()] if prefix else []
        if not isinstance(messages,list): 
            messages = [messages]
        messages = self._create_messages(messages+pref)
        if stream: 
            return self._stream(messages, 
                                prefix=prefix, 
                                max_tokens=max_tokens, 
                                system=system_prompt, 
                                temperature=temperature, 
                                **kwargs)
        response = self.client.messages.create(
            model=self.model, 
            messages=messages,
            max_tokens=max_tokens, 
            system=system_prompt, 
            temperature=temperature, 
            **kwargs)
        
        self._record(response, prefix)

        return self.result

class Claudia:
    """A class representing an AI assistant named Claudia."""
    def __init__(self,
                 model:Optional[str]=None, 
                 system_prompt='', 
                 tools:Optional[list]=None, 
                 tool_choice:Optional[dict]=None):
        """
        Initialize Claudia.
        
        Args:
            model (Optional[str]): claude model to use
            system_prompt (str): Optional system prompt
            tools (Optional[list]): List of tools to make available to Claude
            tool_choice (Optional[dict]): Optionally force use of some tool
        """
        self.client = Client(model)
        self.history = []
        self.system_prompt =  system_prompt
        self.tools = tools
        self.tool_choice = tool_choice

    @property
    def use(self): 
        """Get the current usage statistics."""
        return self.client.use

    def _stream(self, res):
        """
        Stream the response and update the history.
        """
        yield from res
        self.history += self.client.create_toolresponse(self.client.result, ns=self.tools, obj=self)
        
    def __call__(self,
             prompt=None,  
             temperature=0, 
             max_tokens=4096, 
             stream=False,
             prefix='', 
             **kw)-> Union[Message, typing.Generator[str, None, None]]:
        
        if prompt and self.history and nested_idx(self.history, -1, 'role')=='user':
            self() # There is a user request pending, complete it first
 
        if prompt:
            self.history.append(self.client._create_message(prompt))
        if self.tools: 
            kw['tools'] = [get_schema(o) for o in self.tools]
        if self.tool_choice: 
            kw['tool_choice'] = self._create_tool_choice(self.tool_choice)
        response = self.client(
                        messages=self.history, 
                        stream=stream, 
                        prefix=prefix, 
                        system_prompt=self.system_prompt, 
                        temperature=temperature,
                        max_tokens=max_tokens, 
                        **kw,
                    )
        if stream: 
            return self._stream(response)
        self.history += self.client.create_toolresponse(self.client.result, namespace=self.tools, obj=self)
        return response
    
    def _create_tool_choice(self, choose:Union[str, bool, None]) -> dict:
        "Create a `tool_choice` dict that's 'auto' if `choose` is `None`, 'any' if it is True, or 'tool' otherwise"
        if choose:
            if isinstance(choose, str):
                return {"type": "tool", "name": choose}
            return {'type':'any'}
        else:
            return {'type':'auto'}


    #@delegates(self.__call__)
    def tool(self,
            prompt, # Prompt to pass to Claude
            max_loops=10, # Maximum number of tool requests to loop through
            trace_func:Optional[callable]=None, # Function to trace tool using (e.g `print`)
            cont_func:Optional[callable]=noop, # Function that stops loop if returns False
            **kwargs)-> Message:
        "Add prompt `prompt` to dialog and get a response from Claude, automatically following up with `tool_use` messages"
        r = self.__call__(prompt, **kwargs)
        for i in range(max_loops):
            if r.stop_reason!='tool_use': 
                break
            if trace_func: 
                trace_func(r)
            r = self.__call__(**kwargs)
            if not (cont_func or noop)(self.history[-2]): 
                break
        if trace_func: 
            trace_func(r)
        return r

@delegates()
class CoderClaudia(Claudia):
    """
    CoderClaudia is a Claudia client that builds tools to solve problems.
    """

    IMPORTS = 'os, sys, math'
    SYS_PROMPT = f'''You are a knowledgable coding assistant. 
    - Don't do calculations yourself -- you build tools and use code to solve problems.
    - Think carefully about algorithms and computation complexicities.
    - The following modules are pre-imported for `run_code` automatically:
        {IMPORTS}
    - Note that `run_code` interpreter state is *persistent* across calls.
    - If a tool returns `None` report to the user that the attempt was declined and no further progress can be made.
    - Always follow up with verifications and explanations.'''

    def __init__(self, 
                 model: Optional[str] = None, 
                 system_prompt: str = SYS_PROMPT,
                 ask:bool=True, 
                 **kwargs):
        super().__init__(model=model, system_prompt=system_prompt, **kwargs)

        self.ask = ask
        if self.tools is None:
            self.tools = [self.run_code]
        else:
            self.tools = [self.run_code] + self.tools
        self.shell = get_shell()
        self.shell.run_cell(f'import {self.IMPORTS}')
    
    def run(self, prompt:str, **kwargs):
        """Run """
        
        print_colored(f"[User->Claudia]: {prompt}", USER_COLOR)

        self.tool(prompt=prompt, trace_func=self._show_contents, **kwargs)
    
    def run_code(self, code: str) -> str:
        """
        Execute the given code in a persistent session.
        Args:
            code (str): Code to execute in persistent session
        
        Returns:
            str: The result of the code execution
        """
        if self.ask and not self._ask_permission(code):
            return "Code execution was cancelled by user. Rethink the problem and try again."
        try:
            result = self.shell.run_cell(code)
        except Exception as e:
            print_colored( "[Tool->Claudia]: " + str(e), ERROR_COLOR)
            return traceback.format_exc()
        
        if result.stdout:
            res = result.stdout
            print_colored( "[Tool->Claudia]: " + res, RESULT_COLOR)
        else:
            res = result.text
            print_colored( "[Tool->Claudia]: " + res, REMINDER_COLOR)
        return res
    
    def _ask_permission(self, code: str) -> bool:
        """
        Ask for user permission to execute the code.
        
        Args:
            code (str): The code to be executed
        
        Returns:
            bool: True if the user gives permission, False otherwise
        """
        print_colored(f"[Claudia->User]: Do you want to execute the following code?\n", ASK_COLOR)
        print_colored(code, TOOL_COLOR)
        response = input("[User->Tool]: Enter 'y/Y' to execute, any other key to retry: ").lower().strip()
        return response == 'y'
    
    def _show_contents(self, r: Message):
        """Display the contents of a message."""
        for obj in r.content:
            if hasattr(obj,'text'): 
                print_colored( "[Claudia->Tool, User]: " + obj.text, CLAUDE_COLOR)
            name = getattr(obj, 'name', None)
            if name=='run_code':  # in green color
                print_colored( "[Tool]:\n" + obj.input['code'], TOOL_COLOR)
            elif name: 
                print_colored(f'[Claudia]: {obj.name}({obj.input})', REMINDER_COLOR)
            else:
                pass 
        

def test():
    CLAUDE_MODEL = 'claude-3-5-sonnet-20240620'
    coder = CoderClaudia(model=CLAUDE_MODEL, ask=True)

    task = '''find the smallest integer that has its square root larger the meaning of life. hint: use binary search''' 
    _ = coder.run(prompt=task, max_loops=3)
    display(coder.use.__repr__())
    #display(coder.history)


if __name__ == '__main__':
    test()
