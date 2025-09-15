"""This module provides a unified API for querying various large language models."""
import base64
import json
import os
import re
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import anthropic
import requests
from anthropic.types import TextBlock, ThinkingBlock
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from loguru import logger
from openai import OpenAI, RateLimitError
from together import Together
from tqdm import tqdm
import time

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None

class APIQuery:
    """A class to query various large language model APIs."""
    def __init__(self, model, 
                 timeout=9000, 
                 max_tokens=None,
                 api='openai', 
                 max_retries=10,
                 max_retries_inner=5,
                 concurrent_requests=30, 
                 no_system_messages=False,
                 read_cost=1,  
                 write_cost=1,
                 sleep_on_error=60,
                 sleep_after_request=0.1,
                 throw_error_on_failure=False,
                 max_tokens_param="max_tokens",
                 system_prompt=None,
                 developer_message=None,
                 reasoning_effort=None,
                 batch_processing=False,
                 openai_responses=False,
                 max_tool_calls=0,
                 tools=None,
                 **kwargs):
        """Initializes the APIQuery object.

        Args:
            model (str): The name of the model to use.
            timeout (int, optional): The timeout for API requests in seconds. Defaults to 9000.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to None.
            api (str, optional): The API to use. Defaults to 'openai'.
            max_retries (int, optional): The maximum number of retries for a failed query. Defaults to 50.
            concurrent_requests (int, optional): The number of concurrent requests to make. Defaults to 30.
            no_system_messages (bool, optional): Whether to disable system messages. Defaults to False.
            read_cost (int, optional): The cost of reading a token. Defaults to 1.
            write_cost (int, optional): The cost of writing a token. Defaults to 1.
            sleep_on_error (int, optional): The number of seconds to sleep on an error. Defaults to 60.
            sleep_after_request (float, optional): The number of seconds to sleep after a request. Defaults to 0.1.
            throw_error_on_failure (bool, optional): Whether to throw an error on failure. Defaults to False.
            max_tokens_param (str, optional): The name of the max_tokens parameter for the API. Defaults to "max_tokens".
            system_prompt (str, optional): The system prompt to use. Defaults to None.
            developer_message (str, optional): The developer message to use. Defaults to None.
            reasoning_effort (str, optional): The reasoning effort to use. Defaults to None.
            batch_processing (bool, optional): Whether to use batch processing. Defaults to False.
            openai_responses (bool, optional): Whether to use OpenAI responses. Defaults to False.
            max_tool_calls (int, optional): The maximum number of tool calls to make. Defaults to 0.
            tools (list, optional): A list of tools to use. Defaults to None.
            **kwargs: Additional keyword arguments for the API.
        """
        if ("o1" in model or "o3" in model or "o4" in model or "gpt-5" in model) and api == "openai":
            logger.info("Not using system messages for o1/o3/o4 model.")
            no_system_messages = True # o1 model cannot handle system messages
            if not openai_responses:
                max_tokens_param = "max_completion_tokens"
            if "--" in model:
                model, reasoning_effort = model.split("--")
                logger.info(f"Model: {model}, Reasoning effort: {reasoning_effort}")
        if api not in ["anthropic", "openai"] and batch_processing:
            logger.warning("Batch processing is only supported for the Anthropic API and OpenAI API.")
            batch_processing = False
        if openai_responses and not batch_processing:
            max_tokens_param = "max_output_tokens"

        if max_tool_calls > 0 and not openai_responses:
            max_tokens_param = "max_completion_tokens"

        self.kwarg_remover(api, model, kwargs)

        self.model = model
        self.kwargs = kwargs
        if max_tokens is not None:
            self.kwargs[max_tokens_param] = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_retries_inner = max_retries_inner
        self.throw_error_on_failure = throw_error_on_failure
        self.concurrent_requests = concurrent_requests
        self.no_system_messages = no_system_messages
        self.sleep_on_error = sleep_on_error
        self.sleep_after_request = sleep_after_request
        self.read_cost = read_cost
        self.write_cost = write_cost
        self.batch_processing = batch_processing
        self.openai_responses = openai_responses
        self.system_prompt = system_prompt
        self.developer_message = developer_message
        self.max_tool_calls = max_tool_calls
        if max_tokens is not None:
            self.max_tokens_param = max_tokens_param
        if reasoning_effort is not None:
            if not self.openai_responses or self.batch_processing:
                self.kwargs["reasoning_effort"] = reasoning_effort
            elif "reasoning" in self.kwargs:
                self.kwargs["reasoning"]["effort"] = reasoning_effort
            else:
                self.kwargs["reasoning"] = {"effort": reasoning_effort}

        self.tools = tools if tools is not None else []
        self.tool_functions = {tool_desc["function"]["name"]:
                               func for func, tool_desc in self.tools if "function" in tool_desc}
        self.tool_descriptions = [tool_desc for _, tool_desc in self.tools]

        if (self.max_tool_calls == 0 or len(self.tool_descriptions) == 0) and "tool_choice" in self.kwargs:
            del self.kwargs["tool_choice"]
        self.api = api
        self.api_key = None
        self.base_url = None

        self.initialize_api_keys()

        if self.api == "vllm":
            if LLM is None:
                raise ImportError("vllm is not installed. pip install vllm")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            vllm_args = {}

            for p in ("temperature", "top_p", "max_tokens"):
                if p in self.kwargs:
                    vllm_args[p] = self.kwargs.pop(p)
            self.sampling_params = SamplingParams(
                **vllm_args
            )
            self.vllm_model = LLM(
                model=self.model, tensor_parallel_size=4 if 'n_gpus' not in kwargs else kwargs['n_gpus']
            )
            logger.info(f"Loaded local vllm model `{self.model}` with sampling {kwargs}")

    def kwarg_remover(self, api, model, kwargs):
        """Removes kwargs that are not supported by the API or model.

        Args:
            api (str): The API to use.
            model (str): The model to use.
            kwargs (dict): The kwargs to clean.
        """
        if any([kw in model for kw in ["o1", "o3", "o4"]]) and "temperature" in kwargs:
            del kwargs["temperature"]
        for kwarg in ["top_p", "top_k", "temperature"]:
            if kwarg in kwargs and kwargs[kwarg] is None:
                del kwargs[kwarg]
        if (api == "anthropic" and "claude-3-7" in model) or (("o1" in model or "o3" in model) and api == "openai"):
            for kwarg_to_remove in ["top_p", "top_k", "temperature"]:
                if kwarg_to_remove in kwargs:
                    logger.info(f"Removing {kwarg_to_remove} parameter for {model} model.")
                    del kwargs[kwarg_to_remove]

    def initialize_api_keys(self):
        """Initializes the API keys and base URLs for the selected API."""
        if self.api == "xai":
            self.api_key = os.getenv("XAI_API_KEY")
            self.base_url = "https://api.x.ai/v1"
            self.api = "openai"
        elif self.api == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif self.api == "together":
            self.api_key = os.getenv("TOGETHER_API_KEY")
            self.base_url = "https://api.together.xyz/v1"
        elif self.api == "google":
            self.api_key = os.getenv("GOOGLE_API_KEY")
            self.api = "openai"
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        elif self.api == "anthropic":
            if self.max_tool_calls > 0:
                self.api = "openai"
                self.base_url = "https://api.anthropic.com/v1/"
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        elif self.api == "glm":
            self.api_key = os.getenv("GLM_API_KEY")
            self.base_url = "https://open.bigmodel.cn/api/paas/v4/"
            self.api = "openai"
        elif self.api == "deepseek":
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
            self.base_url = "https://api.deepseek.com"
            self.api = "openai"
        elif self.api == "openrouter":
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            self.base_url = "https://openrouter.ai/api/v1"
            if "via_openai" in self.kwargs:
                del self.kwargs["via_openai"]
                self.api = "openai"
        elif self.api == "vllm":
            return
        else:
            raise ValueError(f"API {self.api} not supported.")

        assert self.api_key is not None, f"API key not found."

    def prepare_query(self, query):
        """Prepares a query for the API.

        Args:
            query (tuple): A tuple containing the query and image path.

        Returns:
            tuple: A tuple containing the prepared query and image path.
        """
        query, image_path = query
        if self.no_system_messages:
            # convert system role to user role
            query = [
                {
                    "role": message["role"] if message["role"] != "system" else "developer",
                    "content": message["content"]
                } for message in query
            ]
        
        for message in query:
            if message["role"] == "function_call" and not self.openai_responses:
                message["role"] = "function"
            elif message["role"] == "function_call_output" and not self.openai_responses:
                message["role"] = "tool"
            elif message.get("type", "") in ["function_call", "function_call_output"] and "role" in message and self.openai_responses:
                del message["role"]
                if "content" in message:
                    message["output"] = message["content"]
                    del message["content"]
        return query, image_path
    
    def get_cost(self, response):
        """Calculates the cost of a query.

        Args:
            response (dict): The response from the API.

        Returns:
            float: The cost of the query in dollars.
        """
        cost = response["input_tokens"] * self.read_cost + response["output_tokens"] * self.write_cost
        return cost / (10 ** 6)

    def run_queries(self, queries, no_tqdm=False, allow_tools=True, **kwargs):
        """Runs a list of queries through the API.

        Args:
            queries (list): A list of queries to run.
            no_tqdm (bool, optional): Whether to disable the tqdm progress bar. Defaults to False.
            allow_tools (bool, optional): Whether to allow tool use. Defaults to True.
            **kwargs: Additional keyword arguments for the tokenizer.

        Yields:
            tuple: A tuple containing the index, output, and detailed cost for each query.
        """
        queries_actual = []
        for query in queries:
            if not isinstance(query, tuple):
                queries_actual.append((query, None))
            else:
                queries_actual.append(query)
            if isinstance(queries_actual[-1][0], str):
                queries_actual[-1] = ([{
                    "role": "user",
                    "content": queries_actual[-1][0]
                }], None)
            if self.developer_message is not None and queries_actual[-1][0][0]["role"] != "developer":
                index = 0 if queries_actual[-1][0][0]["role"] != "system" else 1
                queries_actual[-1][0].insert(index, {
                    "role": "developer",
                    "content": self.developer_message
                })

            if self.system_prompt is not None and queries_actual[-1][0][0]["role"] != "system":
                queries_actual[-1][0].insert(0, {
                    "role": "system",
                    "content": self.system_prompt
                })
        
        if not no_tqdm:
            logger.info(f"Running {len(queries_actual)} queries.")

        if self.api == "vllm":
            queries_actual = [self.tokenizer.apply_chat_template(
                                        query[0],
                                        tokenize=False,
                                        add_generation_prompt=True,
                                        **kwargs
                                    ) for query in queries_actual]
            # Bypass threading and batch everything into one local generate
            yield from self._run_vllm_queries(queries_actual)
            return
        
        if self.batch_processing:
            if self.api == "openai":
                processed_results = self.openai_batch_processing(queries_actual)
            else:
                processed_results = self.anthropic_batch_processing(queries_actual)
            for idx, result in enumerate(processed_results):
                if result is None:
                    result = {
                        "output": [{"role": "assistant", "content": ""}],
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }
                detailed_cost = {
                    "cost": self.get_cost(result),
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                }
                yield idx, result["output"], detailed_cost
        else:
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=self.concurrent_requests) as executor:
                future_to_index = {
                    executor.submit(self.run_query_with_retry, query, allow_tools): i
                    for i, query in enumerate(queries_actual)
                }
                
                iterator = as_completed(future_to_index)
                if not no_tqdm:
                    iterator = tqdm(iterator, total=len(future_to_index))

                for future in iterator:
                    idx = future_to_index[future]
                    result = future.result()
                    if result is None:
                        result = {
                            "output": "",
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "time": 0
                        }
                    detailed_cost = {
                        "cost": self.get_cost(result),
                        "input_tokens": result["input_tokens"],
                        "output_tokens": result["output_tokens"],
                        "time": time.time() - start_time
                    }
                    yield idx, result["output"], detailed_cost
    
    def _run_vllm_queries(self, queries_actual):
        """
        Batch all queries into one vllm.generate(...) call, collect final states, yield in order.

        Args:
            queries_actual (list): A list of queries to run.

        Yields:
            tuple: A tuple containing the index, output, and detailed cost for each query.
        """
        tasks = []
        for idx, query in enumerate(queries_actual):
            tasks.append({"id": str(idx), "prompt": query})

        logger.info(f"Running {len(tasks)} queries on local vllm…")
        last_outputs = []
        for batch in self.vllm_model.generate(tasks, sampling_params = self.sampling_params):
            for out in batch.outputs:
                last_outputs.append(out)

        for idx, out in enumerate(last_outputs):
            text = out.text
            inp = getattr(out, "n_input_tokens", 0)
            outp = getattr(out, "n_output_tokens", 0)
            cost = {"cost": (inp*self.read_cost + outp*self.write_cost)/1e6,
                    "input_tokens": inp,
                    "output_tokens": outp}
            yield idx, text, cost

    def run_query_with_retry(self, query, allow_tools=True):
        """Runs a query with retries on failure.

        Args:
            query (tuple): The query to run.
            allow_tools (bool): Whether to allow tool use.

        Returns:
            dict: The result of the query.
        """
        i = 0
        while i < self.max_retries:
            try:
                output = self.run_query(query, allow_tools=allow_tools)
                time.sleep(self.sleep_after_request)
                return output
            except Exception as e:
                logger.error(f"Error: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                time.sleep(self.sleep_on_error)
                # if api error is not due to rate limit, try again
                if "rate limit" not in str(e).lower() and "429" not in str(e):
                    i += 1
                if "violating our usage policy" in str(e).lower():
                    print("Stopping - prompt repeatedly violated usage policy -- ", query)
                    if i > 3:
                        break
                continue
        if self.throw_error_on_failure:
            raise ValueError("Max retries reached.")
        else:
            return {
                "output": "",
                "input_tokens": 0,
                "output_tokens": 0,
            }
    
    def run_query(self, query, allow_tools=True):
        """Runs a single query.

        Args:
            query (tuple): The query to run.
            allow_tools (bool): Whether to allow tool use.

        Returns:
            dict: The result of the query.
        """
        query = self.prepare_query(query)
        if self.api == "openai":
            return self.openai_query_with_tools(query, allow_tools=allow_tools)
        elif self.api == "together":
            return self.openai_query_with_tools(query, is_together=True, allow_tools=allow_tools)
        elif self.api == "anthropic":
            return self.anthropic_query(query)        
        elif self.api == "openrouter":
            if self.max_tool_calls > 0:
                return self.openai_query_with_tools(query)
            else:
                return self.openrouter_query(query)
        
    def postprocess_anthropic_result(self, result):
        """Postprocesses the result from an Anthropic API query.

        Args:
            result: The result from the Anthropic API.

        Returns:
            dict: The postprocessed result.
        """
        output_text = ""

        for content in result.content:
            if isinstance(content, ThinkingBlock):
                output_text += "<think>\n"  + content.thinking + "</think>\n\n"
            elif isinstance(content, TextBlock):
                output_text += content.text
                break
        return {
            "output": [
                {
                    "role": "assistant",
                    "content": output_text
                }
            ],
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
        }
        
    def anthropic_query(self, query):
        """Queries the Anthropic API.

        Args:
            query (tuple): The query to run.

        Returns:
            dict: The result of the query.
        """
        query, image_path = query
        client = anthropic.Anthropic(
            api_key=self.api_key,
            max_retries=0,
            timeout=self.timeout,
        )
        system_message = anthropic.NOT_GIVEN
        if query[0]["role"] == "system":
            system_message = query[0]["content"]
            query = query[1:]
        result = client.messages.create(
            model=self.model,
            messages=query,
            system=system_message,
            **self.kwargs
        )

        return self.postprocess_anthropic_result(result)
    
    def openrouter_query(self, query):
        """Queries the OpenRouter API.

        Args:
            query (tuple): The query to run.

        Returns:
            dict: The result of the query.
        """
        query, image_path = query
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        query_key = "messages"

        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions', 
            headers=headers, 
            json={
                'model': self.model,
                query_key: query,
                "timeout": self.timeout,
                **self.kwargs
            }
        )
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        json_response = response.json()

        if "choices" not in json_response:
            raise Exception(f"Error: {json_response}")

        output = json_response['choices'][0]['message']['content']
        for rk in ["reasoning_content", "reasoning"]:
            if rk in json_response['choices'][0]['message'] and json_response['choices'][0]['message'][rk] is not None:
                output = json_response['choices'][0]['message'][rk] + "</think>" + output
                break
        return {
            "output": [
                {
                    "role": "assistant",
                    "content": output
                }
            ],
            "input_tokens": json_response['usage']['prompt_tokens'],
            "output_tokens": json_response['usage']['completion_tokens'],
        }
    
    def openai_query_responses(self, client, messages, allow_tools=True):
        """Queries the OpenAI API with the responses API.

        Args:
            client: The OpenAI client.
            messages (list): The messages to send.
            allow_tools (bool): Whether to allow tool use.

        Returns:
            dict: The result of the query.
        """
        input_tokens = 0
        output_tokens = 0
        response_tools = []

        for tool_desc in self.tool_descriptions:
            if tool_desc["type"] != "function":
                response_tools.append(tool_desc)
            else:
                response_tools.append({
                    "type": "function",
                    **tool_desc["function"]
                })
        out_msgs = []
        all_out_msgs = []

        if len(response_tools) == 1 and response_tools[0]["type"] == "code_interpreter":
            max_tool_calls = 1
        else:
            max_tool_calls = self.max_tool_calls
        
        if not allow_tools:
            max_tool_calls = 0
        
        current_tool_calls = 0
        for _ in range(max_tool_calls + 1):
            start_tool_calls = current_tool_calls
            response = None
            
            n_tries = 0
            while response is None and n_tries < self.max_retries_inner:
                n_tries += 1
                try:
                    response = client.responses.create(
                        model=self.model,
                        tools=response_tools if current_tool_calls < max_tool_calls else None,
                        input=messages + out_msgs,
                        timeout=self.timeout,
                        **self.kwargs
                    )
                except Exception as e:
                    time.sleep(60)
                    logger.error(f"Got OpenAI error: {e}")
                    continue
            
            if response is None:
                raise ValueError("Max retries reached.")
            
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens

            for out in response.output:
                if out.type == "message":
                    for c in out.content:
                        if c.type == "output_text":
                            out_msgs.append({"role": "assistant", "content": c.text})
                elif out.type == "code_interpreter_call":
                    out_msgs.append({"role": "assistant", "type": "code_interpreter_call", "content": out.code, "id": out.id})
                elif out.type == "function_call":
                    function_name = out.name
                    arguments = json.loads(out.arguments)
                    tool_func = self.tool_functions[function_name]
                    if current_tool_calls > self.max_tool_calls:
                        output = f"Error: Exceeded maximum number of tool calls ({self.max_tool_calls})."
                    else:
                        try:
                            output = tool_func(**arguments)
                        except Exception as e:
                            output = f"Error executing tool {function_name}: {e}"
                    if not isinstance(output, str):
                        additional_cost = output[1]
                        input_tokens += additional_cost["input_tokens"]
                        output_tokens += additional_cost["output_tokens"]
                        output = output[0]
                    current_tool_calls += 1
                    n_execs_left = self.max_tool_calls - current_tool_calls
                    info = f"\n\n### INFO ###\nYou have {n_execs_left} tool executions left."
                    parsed_output = output + info
                    out_msgs.append({
                        "type": "function_call",
                        "call_id": out.call_id,
                        "arguments": out.arguments,
                        "name": out.name,
                    })
                    all_out_msgs.append(out_msgs[-1].copy())
                    if "role" not in all_out_msgs[-1]:
                        all_out_msgs[-1]["role"] = "assistant"
                    out_msgs.append({"type": "function_call_output", 
                                    "call_id": out.call_id,
                                    "output": parsed_output})
                elif out.type != "reasoning":
                    raise ValueError(f"Unknown type {out.type}")
                
                if out.type == "reasoning":
                    summary = "<summary>\n"
                    for thought in out.summary:
                        if thought.text is not None:
                            summary += "<thought>" + "\n" + thought.text + "\n" + "</thought>\n"
                    summary += "</summary>\n"
                    all_out_msgs.append({"role": "assistant", "type": "reasoning", "content": summary, "id": out.id})
                else:
                    all_out_msgs.append(out_msgs[-1].copy())
                    if "role" not in all_out_msgs[-1]:
                        all_out_msgs[-1]["role"] = "assistant"
                        if "content" not in all_out_msgs[-1]:
                            all_out_msgs[-1]["content"] = all_out_msgs[-1].get("output", "")
                            if "output" in all_out_msgs[-1]:
                                del all_out_msgs[-1]["output"]
            if start_tool_calls == current_tool_calls:
                break
        
        if len(all_out_msgs) == 0:
            all_out_msgs.append({"role": "assistant", "content": ""})
        return {
            "output": all_out_msgs,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    
    def get_tokens(self, response):
        """Gets the number of input and output tokens from a response.

        Args:
            response: The response from the API.

        Returns:
            tuple: A tuple containing the number of input and output tokens.
        """
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.total_tokens - response.usage.prompt_tokens
        return input_tokens, output_tokens
    
    def openai_query_no_response(self, client, messages, allow_tools=True):
        """Queries the OpenAI API without the responses API (completions API).

        Args:
            client: The OpenAI client.
            messages (list): The messages to send.
            allow_tools (bool): Whether to allow tool use.

        Returns:
            dict: The result of the query.
        """
        input_tokens = 0
        output_tokens = 0
        output_messages = []
        current_tool_calls = 0
        parsed_output_msgs = []

        max_tool_calls = self.max_tool_calls
        if not allow_tools:
            max_tool_calls = 0

        for it in range(max_tool_calls + 1):
            start_tool_calls = current_tool_calls
            response = None
            n_retries = 0
            while response is None and n_retries < self.max_retries_inner:
                n_retries += 1
                try:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=messages + output_messages,
                        tools=None if current_tool_calls >= max_tool_calls else self.tool_descriptions,
                        timeout=self.timeout,
                        **self.kwargs
                    )
                except Exception as e:
                    logger.info(f"Got OpenAI error: {e}")
                    time.sleep(60)               
                    if isinstance(e, RateLimitError):
                        logger.info("Got OpenAI rate limit error. Sleeping for 60 seconds.")
                        
                        continue
                    else:
                        continue
            if response is None:
                raise ValueError("Max retries reached.")
            input_here, output_here = self.get_tokens(response)
            input_tokens += input_here
            output_tokens += output_here
            output_messages.append(response.choices[0].message)

            msg_content = ""
            if hasattr(response.choices[0].message, 'reasoning') and response.choices[0].message.reasoning:
                msg_content += "<think>" + response.choices[0].message.reasoning + "</think>\n"
            if response.choices[0].message.content is not None:
                msg_content += response.choices[0].message.content            
            if len(msg_content) > 0:
                parsed_output_msgs.append({"role": response.choices[0].message.role, "content": msg_content})

            if not response.choices[0].message.tool_calls:
                break
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                if function_name in self.tool_functions:
                    arguments = json.loads(tool_call.function.arguments)
                    tool_func = self.tool_functions[function_name]
                    if current_tool_calls > max_tool_calls:
                        output = f"Error: Exceeded maximum number of tool calls ({max_tool_calls})."
                    else:
                        output = tool_func(**arguments)
                    if not isinstance(output, str):
                        additional_cost = output[1]
                        input_tokens += additional_cost["input_tokens"]
                        output_tokens += additional_cost["output_tokens"]
                        output = output[0]
                    current_tool_calls += 1
                    n_execs_left = max_tool_calls - current_tool_calls
                    info = f"\n\n### INFO ###\nYou have {n_execs_left} tool executions left."
                    parsed_output = output + info
                    output_messages.append({
                        "role": "tool",
                        "content": parsed_output,
                        "tool_call_id": tool_call.id
                    })
                    parsed_output_msgs.append({
                        "role": "function_call",
                        "tool_name": function_name,
                        "content": tool_call.function.arguments,
                        "tool_call_id": tool_call.id
                    })
                    parsed_output_msgs.append({
                        "role": "function_call_output",
                        "content": parsed_output,
                        "tool_call_id": tool_call.id
                    })

            if start_tool_calls == current_tool_calls:
                break

        if len(parsed_output_msgs) == 0:
            parsed_output_msgs.append({"role": "assistant", "content": ""})
        return {
            "output": parsed_output_msgs,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


    def openai_query_with_tools(self, query, is_together=False, allow_tools=True):
        """Queries the OpenAI API with tools.

        Args:
            query (tuple): The query to run.
            is_together (bool, optional): Whether to use the Together API. Defaults to False.
            allow_tools (bool, optional): Whether to allow tool use. Defaults to True.

        Returns:
            dict: The result of the query.
        """
        if is_together:
            client = Together()
        else:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url, 
                            timeout=self.timeout,  max_retries=0)
        messages, image_path = query

        if self.openai_responses:
            return self.openai_query_responses(client, messages, allow_tools=allow_tools) 
        else:
            return self.openai_query_no_response(client, messages, allow_tools=allow_tools)        
            
        
    def openai_batch_processing(self, queries, error_repetition=0):
        """Processes a batch of queries using the OpenAI API.

        Args:
            queries (list): A list of queries to run.
            error_repetition (int, optional): The number of times to repeat a query on error. Defaults to 0.

        Returns:
            list: A list of results.
        """
        if error_repetition >= self.max_retries:
            return [
                {
                    "output": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                } for _ in range(len(queries))
            ]
        text_queries = [query[0] for query in queries]
        jsonl_queries = []

        for i, query in enumerate(text_queries):
            request = {
                "custom_id": f"apiquery-{i}",
                "method": "POST", 
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": query,
                    **self.kwargs
                }
            }
            jsonl_queries.append(request)
        
        client = OpenAI(api_key=self.api_key, base_url=self.base_url, 
                            max_retries=0)
        
        # create temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        with open(tmp.name, "wb") as f:
            for i, query in enumerate(jsonl_queries):
                f.write(json.dumps(query).encode("utf-8"))
                f.write(b"\n")

        batch_input_file = client.files.create(
            file=open(tmp.name, "rb"),
            purpose="batch"
        )

        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        # close tmp file
        tmp.close()

        logger.info(f"Running {len(queries)} queries with batch ID {batch.id} using file with File ID {batch_input_file.id}.")

        request_counts = dict(batch.request_counts)
        
        while True:
            try:
                batch = client.batches.retrieve(batch.id)
            except Exception as e:
                logger.warning(f"Error connecting to OpenAI. Retrying in 10s.")
                pass
            if any([request_counts[key] != dict(batch.request_counts)[key] for key in request_counts]):
                request_counts = dict(batch.request_counts)
                logger.info(f"Completed Requests Progress: {request_counts['completed']}/{len(queries)}. Errors: {request_counts['failed']}/{len(queries)}")
            if batch.status == "completed":
                break
            time.sleep(10)
        
        outputs = [None for _ in range(len(queries))]
        repeat_indices = []

        if batch.output_file_id is None:
            return outputs
        while True:
            try:
                file_response = client.files.content(file_id=batch.output_file_id)
                break
            except Exception as e:
                logger.error(f"Error connecting to OpenAI: {e}. Retrying in 10 seconds.")
                time.sleep(10)
                continue

        json_response = []
        for line in file_response.iter_lines():
            json_response.append(json.loads(line))

        for result in json_response:
            index = int(result["custom_id"].split("-")[-1])
            if result["response"]["status_code"] != 200:
                repeat_indices.append(index)
                logger.error(f"Error: {result['response']['status_code']}")
            else:
                try:
                    outputs[index] = {
                        "output": [
                            {
                                "role": "assistant",
                                "content": result["response"]["body"]["choices"][0]["message"]["content"]
                            }
                        ],
                        "input_tokens": result["response"]["body"]["usage"]["prompt_tokens"],
                        "output_tokens": result["response"]["body"]["usage"]["completion_tokens"],
                    }
                except Exception as e:
                    logger.error(f"Error: {e}")
                    repeat_indices.append(index)
        
        for i in range(len(outputs)):
            if outputs[i] is None:
                repeat_indices.append(i)
        if len(repeat_indices) > 0:
            logger.info(f"Repeating {len(repeat_indices)} queries.")
            repeat_queries = [queries[i] for i in repeat_indices]
            repeat_outputs = self.openai_batch_processing(repeat_queries, error_repetition + 1)
            for i, output in zip(repeat_indices, repeat_outputs):
                outputs[i] = output
        
        return outputs
    
    def anthropic_batch_processing(self, queries, error_repetition=0):
        """Processes a batch of queries using the Anthropic API.

        Args:
            queries (list): A list of queries to run.
            error_repetition (int, optional): The number of times to repeat a query on error. Defaults to 0.

        Returns:
            list: A list of results.
        """
        if error_repetition >= self.max_retries:
            return [
                {
                    "output": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                } for _ in range(len(queries))
            ]

        text_queries = [query[0] for query in queries]
        client = anthropic.Anthropic(
            api_key=self.api_key,
            max_retries=0,
        )

        requests = []

        for i, text_query in enumerate(text_queries):
            kwargs_here = self.kwargs.copy()
            if text_query[0]["role"] == "system":
                kwargs_here["system"] = text_query[0]["content"]
                text_query = text_query[1:]
            
            request = Request(
                custom_id=f"apiquery-{i}",
                params=MessageCreateParamsNonStreaming(
                    model=self.model,
                    messages=text_query,
                    **kwargs_here
                )
            )
            requests.append(request)
        
        message_batch = client.messages.batches.create(requests=requests)

        logger.info(f"Running {len(queries)} queries with batch ID {message_batch.id}")

        current_request_counts = dict(message_batch.request_counts)

        while True:
            try:
                message_batch = client.messages.batches.retrieve(
                    message_batch_id=message_batch.id,
                )
            except:
                logger.warning(f"Error connecting to Anthropic. Retrying in 10s.")
                pass
            if any([current_request_counts[key] != dict(message_batch.request_counts)[key] 
                    for key in current_request_counts]):
                current_request_counts = dict(message_batch.request_counts)
                error_sum = sum([current_request_counts[key] for key in current_request_counts if "succeeded" != key])
                logger.info(f"Succeeded Requests Progress: {current_request_counts['succeeded']}/{len(queries)}. Errors: {error_sum}")
            if message_batch.processing_status == "ended":
                break
            time.sleep(10)
        
        outputs = []
        repeat_indices = []

        while True:
            try:
                results = client.messages.batches.results(
                    message_batch_id=message_batch.id,
                )
                break
            except Exception as e:
                logger.error(f"Error connecting to Anthropic: {e}. Retrying in 10 seconds.")
                time.sleep(10)

        for i, result in enumerate(results):  
            if result.result.type == "succeeded":
                outputs.append(self.postprocess_anthropic_result(result.result.message))
            else:
                outputs.append(None)
                repeat_indices.append(i)
                if result.result.type == "errored":
                    logger.error(result.result.error)

        if len(repeat_indices) > 0:
            logger.info(f"Repeating {len(repeat_indices)} queries.")
            repeat_queries = [queries[i] for i in repeat_indices]
            repeat_outputs = self.anthropic_batch_processing(repeat_queries, error_repetition + 1)
            for i, output in zip(repeat_indices, repeat_outputs):
                outputs[i] = output
        
        return outputs
