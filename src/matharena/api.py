from loguru import logger
import re
import os
from tqdm import tqdm
from google import genai
from openai import OpenAI
from together import Together
import anthropic
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import base64
import subprocess
import requests

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class APIQuery:
    def __init__(self, model, 
                 timeout=500, 
                 temperature=0, 
                 max_tokens=None,
                 api='openai', 
                 max_retries=50,
                 concurrent_requests=30, 
                 is_chat=True,
                 no_system_messages=False,
                 read_cost=1,  
                 write_cost=1,
                 sleep_on_error=60,
                 sleep_after_request=0.1,
                 throw_error_on_failure=False,
                 max_tokens_param="max_tokens",
                 reasoning_effort=None,
                 **kwargs):
        if "think" in model and api == "google":
            is_chat = False # think model cannot handle chat
            max_tokens_param = "max_output_tokens"
        if ("o1" in model or "o3" in model) and api == "openai":
            no_system_messages = True # o1 model cannot handle system messages
            max_tokens_param = "max_completion_tokens"
            if "top_p" in kwargs:
                del kwargs["top_p"]
            if "top_k" in kwargs:
                del kwargs["top_k"]
            if "--" in model:
                model, reasoning_effort = model.split("--")

        if api == "anthropic":
            max_tokens = min(8192, max_tokens)
        if api == "deepseek":
            max_tokens = min(8192, max_tokens)

        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs
        self.kwargs[max_tokens_param] = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.throw_error_on_failure = throw_error_on_failure
        self.concurrent_requests = concurrent_requests
        self.is_chat = is_chat
        self.no_system_messages = no_system_messages
        self.sleep_on_error = sleep_on_error
        self.sleep_after_request = sleep_after_request
        self.read_cost = read_cost
        self.write_cost = write_cost

        if max_tokens is not None:
            self.max_tokens_param = max_tokens_param
        if reasoning_effort is not None:
            self.kwargs["reasoning_effort"] = reasoning_effort

        self.api = api
        self.api_key = None
        self.base_url = None

        self.initialize_api_keys()

    def initialize_api_keys(self):
        if self.api == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif self.api == "together":
            self.api_key = os.getenv("TOGETHER_API_KEY")
            self.base_url = "https://api.together.xyz/v1"
        elif self.api == "google":
            self.api_key = os.getenv("GOOGLE_API_KEY")
            if not "think" in self.model:
                self.api = "openai"
                self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        elif self.api == "anthropic":
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        elif self.api == "hyperbolic":
            self.api_key = os.getenv("HYPERBOLIC_API_KEY")
            self.base_url = "https://api.hyperbolic.xyz/v1"
            self.api = "openai"
        elif self.api == 'sambanova':
            self.api_key = os.getenv("SAMBA_API_KEY")
            self.base_url = "https://api.sambanova.ai/v1"
            self.api = "openai"
        elif self.api == "deepseek":
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
            self.base_url = "https://api.deepseek.com"
            self.api = "openai"
        elif self.api == "openrouter":
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            self.base_url = "https://openrouter.ai/api/v1"
            self.api = "openai"
        elif self.api == "fireworks":
            self.api_key = os.getenv("FIREWORKS_API_KEY")
            self.base_url = "https://api.fireworks.ai/inference/v1"
            self.api = "openai"
        elif self.api == "vllm":
            self.api_key = "token-abc123"
            self.api = "openai"
            self.base_url = f"http://localhost:8000/v1"
            # command = f"vllm serve {self.model} --dtype auto --api-key token-abc123"
            # Launch the command in the background.
            # subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Poll the server until it's running.
        else:
            raise ValueError(f"API {self.api} not supported.")
        assert self.api_key is not None, f"API key not found."

    def prepare_query(self, query):
        query, image_path = query
        if not self.is_chat:
            output_query = query[0]["content"]
            for message in query:
                output_query += f"\n\n{'=' * 20}{message['role']}{'=' * 20}\n\n{message['content']}"
            return output_query, image_path
        elif self.no_system_messages:
            # convert system role to user role
            query = [{
                "role": message["role"] if message["role"] != "system" else "user",
                "content": message["content"]
            } for message in query]
        return query, image_path
    
    def get_cost(self, response):
        cost = response["input_tokens"] * self.read_cost + response["output_tokens"] * self.write_cost
        return cost / (10 ** 6)

    def run_queries(self, queries):
        if self.api == "vllm":
            while True:
                try:
                    response = requests.get(f"{self.base_url}", timeout=1)
                    if response.status_code == 401: # unauthorized, because no api key here
                        break
                except Exception:
                    pass
                time.sleep(5)
                logger.info("Waiting for VLLM server to start...")
            logger.info("VLLM server started.")

        logger.info(f"Running {len(queries)} queries.")
        with ThreadPoolExecutor(max_workers=self.concurrent_requests) as executor:
            future_to_index = {
                executor.submit(self.run_query_with_retry, query): i
                for i, query in enumerate(queries)
            }
            for future in tqdm(as_completed(future_to_index), total=len(future_to_index)):
                idx = future_to_index[future]
                result = future.result()
                detailed_cost = {
                    "cost": self.get_cost(result),
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                }
                yield idx, result["output"], detailed_cost

        # detailed_cost = [
        #     {
        #         "cost": self.get_cost(result),
        #         "input_tokens": result["input_tokens"],
        #         "output_tokens": result["output_tokens"],
        #     }
        #     for result in results
        # ]
        # return [result['output'] for result in results], detailed_cost
    
    def run_query_with_retry(self, query):
        i = 0
        while i < self.max_retries:
            try:
                output = self.run_query(query)
                time.sleep(self.sleep_after_request)
                return output
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(self.sleep_on_error)
                # if api error is not due to rate limit, try again
                if "rate limit" not in str(e).lower() and "429" not in str(e):
                    i += 1
                continue
        if self.throw_error_on_failure:
            raise ValueError("Max retries reached.")
        else:
            return {
                "output": "",
                "input_tokens": 0,
                "output_tokens": 0,
            }
    
    def run_query(self, query):
        query = self.prepare_query(query)
        if self.api == "openai":
            return self.openai_query(query)
        elif self.api == "together":
            return self.together_query(query)
        elif self.api == "google":
            return self.google_query(query)
        elif self.api == "anthropic":
            return self.anthropic_query(query)
        
    def anthropic_query(self, query):
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
            temperature=self.temperature,
            **self.kwargs
        )
        return {
            "output": result.content[0].text,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
        }
    
    def google_query(self, query):
        client = genai.Client(api_key=self.api_key, http_options={'api_version':'v1alpha'})
        query, image_path = query
        if image_path is not None:
            file = client.files.upload(file=image_path)
            query = [query, file]
        config = {
            'temperature': self.temperature,
            # 'max_output_tokens': self.kwargs[self.max_tokens_param],
        }
        # if "think" in self.model:
        #     config['thinking_config'] = {'include_thoughts': True}
        response = client.models.generate_content(
            model=self.model,
            contents=query,
            # config=config,
        )
        return {
            "output": "\n\n".join([response.candidates[0].content.parts[i].text 
                                   for i in range(len(response.candidates[0].content.parts))]),
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
        }

    def together_query(self, query):
        client = Together()
        query, image_path = query
        response = client.chat.completions.create(
            model=self.model,
            messages=query,
            temperature=self.temperature,
            **self.kwargs
        )
        output = response.choices[0].message.content
        if hasattr(response.choices[0].message, "reasoning_content"):
            output = response.choices[0].message.reasoning_content + "\n\n" + output
        return {
            "output": output,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

    
    def openai_query(self, query):
        client = OpenAI(api_key=self.api_key, base_url=self.base_url, 
                        timeout=self.timeout, max_retries=0)
        query, image_path = query
        if image_path is not None:
            base64_image = encode_image(image_path)
            query.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]})

        if "o3" in self.model or "o1" in self.model:
            response = client.chat.completions.create(
                model=self.model,
                messages=query,
                **self.kwargs
            )
        else:
            response = client.chat.completions.create(
                model=self.model,
                messages=query,
                temperature=self.temperature,
                **self.kwargs
            )
        # print(response)
        output = response.choices[0].message.content
        if hasattr(response.choices[0].message, "reasoning_content") and \
            response.choices[0].message.reasoning_content is not None:
            output = response.choices[0].message.reasoning_content + "\n\n" + output
        return {
            "output": output,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
