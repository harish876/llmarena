"""This module contains the main runner for conducting experiments."""
import csv
import json
import os

import sympy
import yaml
from datasets import load_dataset
from loguru import logger
from transformers import AutoTokenizer

from matharena.api import APIQuery
from matharena.code_execution import execute_code
from matharena.cot_solver import CoTSolver
from matharena.parser import WarningType, check_answers, extract_answer, parse_answer
from matharena.possible_issues import (
    check_all_numbers,
    check_number_proximity_any_order,
    check_output_length,
)
from matharena.solveragent import SolverAgent


def validate_messages(messages, competition, detailed_cost):
    """Validates a list of messages.

    Args:
        messages (list): A list of messages.
        competition (str): The name of the competition.
        detailed_cost (dict): The detailed cost of the messages.

    Returns:
        bool: True if the messages are valid, False otherwise.
    """
    # if "euler" in competition:
    #     return True
    
    # if detailed_cost["input_tokens"] > 10000:
    #     return False # OpenAI is caching stuff and failing at it... 
    
    if not any([message['role'] == 'assistant' for message in messages]):
        return False
    
    for message in messages[::-1]:
        if message['role'] == 'assistant' and message.get("type", "") != "reasoning" and len(message['content']) == 0:
            return False
        if message['role'] == 'assistant' and message.get("type", "") != "reasoning" and len(message['content']) > 0:
            break
        
    return True

def _load_model_and_competition_configs(model_config, competition, competition_config_folder):
    """Loads model and competition configurations.

    Args:
        model_config (dict): The model configuration.
        competition (str): The name of the competition.
        competition_config_folder (str): The path to the competition configuration folder.

    Returns:
        tuple: A tuple containing the model configuration, competition configuration,
            agent name, and agent configuration.
    """
    agent = None
    agent_config = None
    if model_config.get("type") == "agent":
        agent_config_path = os.path.join("configs", model_config["agent_config"] + ".yaml")
        with open(agent_config_path, "r") as f:
            agent_config = yaml.safe_load(f)
        
        human_readable_id = model_config["human_readable_id"]
        model_config_path = os.path.join("configs", model_config["model_config"] + ".yaml")
        n = model_config["n"]
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)
        model_config["human_readable_id"] = human_readable_id
        model_config["n"] = n
        if "concurrent_requests" in model_config:
            agent_config["n_threads"] = model_config["concurrent_requests"]
        
        agent = agent_config.pop("type")
    
    competition_config_path = f"{competition_config_folder}/{competition}.yaml"
    with open(competition_config_path, "r") as f:
        competition_config = yaml.safe_load(f)
        
    return model_config, competition_config, agent, agent_config

def _prepare_tools(competition_config, model_config):
    """Prepares tools for the agent.

    Args:
        competition_config (dict): The competition configuration.
        model_config (dict): The model configuration.

    Returns:
        list: A list of tools.
    """
    tool_descriptions = competition_config.get("tools", [])
    TOOL_FUNCTIONS = {"execute_code": execute_code}
    tools = []
    for tool_desc in tool_descriptions:
        if model_config.get("openai_responses", False) and tool_desc.get("openai_responses_tool"):
            tools.append((None, tool_desc["openai_responses_tool"]))
        else:
            tool_desc = tool_desc["tool"]
            func_name = tool_desc["function"]["name"]
            if func_name in TOOL_FUNCTIONS:
                tools.append((TOOL_FUNCTIONS[func_name], tool_desc))
    return tools

def _prepare_api_kwargs(model_config, competition_config, tools):
    """Prepares kwargs for the API caller.

    Args:
        model_config (dict): The model configuration.
        competition_config (dict): The competition configuration.
        tools (list): A list of tools.

    Returns:
        dict: A dictionary of kwargs for the API caller.
    """
    kwargs = model_config.copy()
    kwargs.pop("model")
    kwargs.pop("n", None)
    kwargs.pop("api")
    kwargs.pop("human_readable_id")
    kwargs.pop("tokenizer_kwargs", None)
    kwargs.pop("date", None)
    
    if "max_tool_calls" in competition_config:
        kwargs["max_tool_calls"] = competition_config["max_tool_calls"]
        
    kwargs["tools"] = tools
    return kwargs

def _load_problems(competition_config):
    """Loads problems for the competition.

    Args:
        competition_config (dict): The competition configuration.

    Returns:
        list: A list of problems.
    """
    dataset_path = competition_config["dataset_path"]

    if os.path.exists(dataset_path):
        answers_path = os.path.join(dataset_path, "answers.csv")
        source_path = os.path.join(dataset_path, "source.csv")
        type_path = os.path.join(dataset_path, "problem_types.csv")
        problems = []
        
        problem_types = None
        if os.path.exists(type_path):
            with open(type_path, "r") as f:
                problem_types_reader = csv.DictReader(f)
                problem_types = {int(row["id"]): row["type"] for row in problem_types_reader}
                for problem_id in problem_types:
                    problem_types[problem_id] = problem_types[problem_id].replace('"', "").replace("[", "").replace("]", "").split(',')

        with open(answers_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                id_val = int(row["id"])
                problem_path = os.path.join(dataset_path, "problems", f"{id_val}.tex")
                with open(problem_path, "r") as f_problem:
                    problem_content = f_problem.read()
                
                problem_type_val = None
                if problem_types and id_val in problem_types:
                    problem_type_val = problem_types[id_val]

                problems.append({
                    "problem_idx": id_val, 
                    "problem": problem_content, 
                    "answer": row["answer"], 
                    "problem_type": problem_type_val
                })
        
        if os.path.exists(source_path):
            with open(source_path, "r") as f:
                source_reader = csv.DictReader(f)
                # Create a mapping from id to source for efficient lookup
                source_map = {int(row["id"]): row["source"] for row in source_reader}
                for p in problems:
                    if p['problem_idx'] in source_map:
                        p["source"] = source_map[p['problem_idx']]

    else:
        problems = load_dataset(dataset_path, split="train").to_list()

    return sorted(problems, key=lambda x: x["problem_idx"])

def _process_existing_results(output_file, competition, recompute_tokens, model_config, api, skip_all):
    """Processes existing results for a problem.

    Args:
        output_file (str): The path to the output file.
        competition (str): The name of the competition.
        recompute_tokens (bool): Whether to recompute tokens.
        model_config (dict): The model configuration.
        api (str): The API to use.
        skip_all (bool): Whether to skip all problems.

    Returns:
        tuple: A tuple containing the messages, detailed costs, and histories.
    """
    with open(output_file, 'r') as f:
        data_file = json.load(f)
        
    messages = data_file["messages"]
    histories = data_file.get("history", [None] * len(messages))

    if recompute_tokens and api == 'vllm':
        tokenizer = AutoTokenizer.from_pretrained(model_config["model"])
        new_detailed_costs = []
        for msg_list in messages:
            input_tokens = sum(len(tokenizer.encode(m["content"])) for m in msg_list if m["role"] != "assistant")
            output_tokens = sum(len(tokenizer.encode(m["content"])) for m in msg_list if m["role"] == "assistant")
            cost = (input_tokens * model_config.get('read_cost', 0) + output_tokens * model_config.get('write_cost', 0)) / 1_000_000
            new_detailed_costs.append({"cost": cost, "input_tokens": input_tokens, "output_tokens": output_tokens})
        detailed_costs = new_detailed_costs
    else:
        if "detailed_costs" in data_file:
            detailed_costs = data_file["detailed_costs"]
        else:
            cost = data_file.get("cost", {})
            detailed_costs = [{"cost": cost.get("cost", 0) if i == 0 else 0,
                               "input_tokens": cost.get("input_tokens", 0) if i == 0 else 0,
                               "output_tokens": cost.get("output_tokens", 0) if i == 0 else 0}
                              for i in range(len(messages))]

    valid_indices = [i for i, (msg, cost) in enumerate(zip(messages, detailed_costs)) if validate_messages(msg, competition, cost) or skip_all]
    
    messages = [messages[i] for i in valid_indices]
    detailed_costs = [detailed_costs[i] for i in valid_indices]
    histories = [histories[i] for i in valid_indices]
    
    return messages, detailed_costs, histories

def _prepare_batches(problems, output_dir, n, skip_existing, recompute_tokens, model_config, competition, skip_all, prompt_template, competition_config):
    """Prepares batches for the agent, skipping existing problems if necessary.

    Args:
        problems (list): A list of problems.
        output_dir (str): The path to the output directory.
        n (int): The number of samples to generate for each problem.
        skip_existing (bool): Whether to skip existing problems.
        recompute_tokens (bool): Whether to recompute tokens.
        model_config (dict): The model configuration.
        competition (str): The name of the competition.
        skip_all (bool): Whether to skip all problems.
        prompt_template (str): The prompt template.
        competition_config (dict): The competition configuration.

    Returns:
        tuple: A tuple containing the batch prompts, batch index to problem index mapping,
            all messages per problem, detailed costs per problem, and all histories per problem.
    """
    batch_prompts = []
    batch_idx_to_problem_idx = {}
    all_messages_per_problem = {i: [] for i in range(len(problems))}
    detailed_costs_per_problem = {i: [] for i in range(len(problems))}
    all_histories_per_problem = {i: [] for i in range(len(problems))}
    final_answer_comp = competition_config.get("final_answer", True)

    for i, problem in enumerate(problems):
        problem_id = problem["problem_idx"]
        output_file = os.path.join(output_dir, f"{problem_id}.json")
        if skip_existing and os.path.exists(output_file):
            messages, detailed_costs, histories = _process_existing_results(output_file, competition, recompute_tokens, model_config, model_config["api"], skip_all)
            
            detailed_costs_per_problem[i] = detailed_costs
            all_messages_per_problem[i] = messages
            all_histories_per_problem[i] = histories
            
            logger.info(f"Skipping problem: {problem_id} ({len(messages)} times)")
            if len(messages) == n or skip_all:
                calculate_problem_results(model_config, problem, output_dir, messages,
                                          detailed_costs, histories, i, competition_config.get("strict_parsing"),
                                          final_answer=final_answer_comp)
                continue

        problem_statement = problem["problem"]
        problem_prompt = prompt_template.format(problem_statement=problem_statement)
        for _ in range(n - len(all_messages_per_problem[i])):
            batch_idx_to_problem_idx[len(batch_prompts)] = i
            batch_prompts.append((problem_prompt, None))
            
    return batch_prompts, batch_idx_to_problem_idx, all_messages_per_problem, detailed_costs_per_problem, all_histories_per_problem

def _initialize_agent(agent, model, api, api_kwargs, agent_config):
    """Initializes the correct agent.

    Args:
        agent (str): The name of the agent.
        model (str): The name of the model.
        api (str): The API to use.
        api_kwargs (dict): The kwargs for the API caller.
        agent_config (dict): The agent configuration.

    Returns:
        object: The initialized agent.
    """
    if not agent:
        api_caller = APIQuery(model=model, api=api, **api_kwargs)
        return CoTSolver(querier=api_caller)
    
    agent_params = {"model": model, "api": api, **api_kwargs}
    
    if agent == "solver":
        return SolverAgent(agent_params, **agent_config)
    else:
        raise ValueError(f"Unknown agent type: {agent}")
    
def reprompt(messages, problem, querier, prompt_template, no_boxed_prompt):
    """Reprompts the model if no answer was found.

    Args:
        messages (list): The messages to send.
        problem (dict): The problem.
        querier (object): The API caller.
        prompt_template (str): The prompt template.
        no_boxed_prompt (str): The no boxed prompt.

    Returns:
        tuple: A tuple containing the new messages and the extra cost.
    """
    logger.info("No answer found, reprompting the model")
    problem_statement = problem["problem"]
    new_prompt = prompt_template.format(problem_statement=problem_statement)
    new_messages = [{"role": "user", "content": new_prompt}] + messages + [{"role": "user", "content": no_boxed_prompt}]
    _, response, detailed_cost = list(querier.run_queries([new_messages], allow_tools=False, no_tqdm=True))[0]
    messages.append({"role": "user", "content": no_boxed_prompt})
    if len(response) > 0:
        messages.extend(response)
    else:
        messages.append({"role": "assistant", "content": "\\boxed{None}"})
    return messages, detailed_cost


def _run_agent_and_process_results(cot_solver, batch_prompts, batch_idx_to_problem_idx, problems,
                                   all_messages_per_problem, detailed_costs_per_problem, all_histories_per_problem,
                                   n, model_config, output_dir, competition_config, tokenizer_kwargs):
    """Runs the agent and processes the results.

    Args:
        cot_solver (object): The agent.
        batch_prompts (list): A list of batch prompts.
        batch_idx_to_problem_idx (dict): A mapping from batch index to problem index.
        problems (list): A list of problems.
        all_messages_per_problem (dict): A dictionary of all messages per problem.
        detailed_costs_per_problem (dict): A dictionary of detailed costs per problem.
        all_histories_per_problem (dict): A dictionary of all histories per problem.
        n (int): The number of samples to generate for each problem.
        model_config (dict): The model configuration.
        output_dir (str): The path to the output directory.
        competition_config (dict): The competition configuration.
        tokenizer_kwargs (dict): The tokenizer kwargs.
    """
    if not batch_prompts:
        return

    final_answer_comp = competition_config.get("final_answer", True)
    
    for idx, messages, detailed_cost, history in cot_solver.solve(batch_prompts, **tokenizer_kwargs):
        problem_idx = batch_idx_to_problem_idx[idx]
        problem = problems[problem_idx]
        prompt_template = f"{competition_config['instruction']}\n\n" + "{problem_statement}"
        if final_answer_comp and extract_answer(messages[-1]["content"], True)[0] is None and (len(messages[-1]["content"]) > 0 or len(messages) > 1):
            # essentially no answer found, ask the model to run again
            messages, extra_cost = reprompt(
                messages,
                problem,
                cot_solver.querier,
                prompt_template,
                competition_config.get("no_boxed_prompt", "You did not follow the formatting instructions. Please provide your final answer within \\boxed{}. If you did not find the answer, please use \\boxed{None}."),
            )
            detailed_cost["cost"] += extra_cost["cost"]
            detailed_cost["input_tokens"] += extra_cost["input_tokens"]
            detailed_cost["output_tokens"] += extra_cost["output_tokens"]

        messages = [{"role": "user", "content": prompt_template.format(problem_statement=problem["problem"])}] + messages
        all_messages_per_problem[problem_idx].append(messages)
        detailed_costs_per_problem[problem_idx].append(detailed_cost)
        all_histories_per_problem[problem_idx].append(history)

        log_this = len(all_messages_per_problem[problem_idx]) == n
        calculate_problem_results(model_config, problem, output_dir,
                                all_messages_per_problem[problem_idx],
                                detailed_costs_per_problem[problem_idx],
                                all_histories_per_problem[problem_idx],
                                problem_idx, competition_config.get("strict_parsing"),
                                final_answer=final_answer_comp, log_this=log_this)

def run(model_config, config_path, competition, skip_existing=False, output_folder="outputs", 
        competition_config_folder="competition_configs", skip_all=False, recompute_tokens=False):
    """Runs the experiment.

    Args:
        model_config (dict): The model configuration.
        config_path (str): The path to the model configuration file.
        competition (str): The name of the competition.
        skip_existing (bool, optional): Whether to skip existing problems. Defaults to False.
        output_folder (str, optional): The path to the output folder. Defaults to "outputs".
        competition_config_folder (str, optional): The path to the competition configuration folder.
            Defaults to "competition_configs".
        skip_all (bool, optional): Whether to skip all problems. Defaults to False.
        recompute_tokens (bool, optional): Whether to recompute tokens. Defaults to False.
    """
    
    model_config, competition_config, agent, agent_config = _load_model_and_competition_configs(
        model_config, competition, competition_config_folder
    )
    
    n = model_config["n"]
    model = model_config["model"]
    api = model_config["api"]
    tokenizer_kwargs = model_config.get("tokenizer_kwargs", {})

    tools = _prepare_tools(competition_config, model_config)
    api_kwargs = _prepare_api_kwargs(model_config, competition_config, tools)

    logger.info(f"New run, model: {model}, competition: {competition}")

    problems = _load_problems(competition_config)
    
    output_dir = os.path.join(f"{output_folder}/{competition}/", config_path.replace(".yaml", ""))
    os.makedirs(output_dir, exist_ok=True)

    prompt_template = f"{competition_config['instruction']}\n\n" + "{problem_statement}"

    batch_prompts, batch_idx_to_problem_idx, all_messages_per_problem, detailed_costs_per_problem, all_histories_per_problem = _prepare_batches(
        problems, output_dir, n, skip_existing, recompute_tokens, model_config, competition, skip_all, prompt_template, competition_config
    )

    if not batch_prompts:
        logger.info("No problems to run.")
        return

    logger.info("Collected all queries, now running")

    cot_solver = _initialize_agent(agent, model, api, api_kwargs, agent_config)

    _run_agent_and_process_results(
        cot_solver, batch_prompts, batch_idx_to_problem_idx, problems,
        all_messages_per_problem, detailed_costs_per_problem, all_histories_per_problem,
        n, model_config, output_dir, competition_config, tokenizer_kwargs
    )

def safe_str_int(x, max_digits=4300):
    """Converts an integer to a string, handling large integers.

    Args:
        x: The integer to convert.
        max_digits (int, optional): The maximum number of digits to display. Defaults to 4300.

    Returns:
        str: The string representation of the integer.
    """
    s = str(x)
    if len(s) > max_digits:
        return f"{s[:20]}...({len(s)} digits)...{s[-20:]}"
    return s

def calculate_problem_results(model_config, problem, output_dir, messages_problem, 
                              costs_problem, histories_problem, problem_idx, strict_parsing, 
                              final_answer=True, log_this=True):
    """Calculates and saves the results for a problem.

    Args:
        model_config (dict): The model configuration.
        problem (dict): The problem.
        output_dir (str): The path to the output directory.
        messages_problem (list): A list of messages for the problem.
        costs_problem (list): A list of costs for the problem.
        histories_problem (list): A list of histories for the problem.
        problem_idx (int): The index of the problem.
        strict_parsing (bool): Whether to use strict parsing.
        final_answer (bool, optional): Whether to expect a final answer. Defaults to True.
        log_this (bool, optional): Whether to log the results. Defaults to True.
    """
    problem_id = problem["problem_idx"]

    problem_statement = problem["problem"]
    list_answer = "," in str(problem["answer"]) if final_answer else False

    if final_answer:
        gold_answer, _ = parse_answer(str(problem["answer"]), list_answer=list_answer)
    else:
        gold_answer = None
    output_file = os.path.join(output_dir, f"{problem_id}.json")
    n = len(messages_problem)
    answers = []
    warnings = []
    corrects = []
    for j in range(n):
        if final_answer:
            model_answer = messages_problem[j][-1]["content"]
            if not isinstance(model_answer, str):
                raise ValueError(f"Model answer is not a string: {model_answer}")
            model_answer, warning = extract_answer(model_answer, strict_parsing, True, list_answer)
            is_correct = check_answers(model_answer, gold_answer)
            if not is_correct and check_output_length(costs_problem[j]["output_tokens"]):
                logger.warning(f"Model output length {costs_problem[j]['output_tokens']} is of the form 10**k * 2**n. This might indicate it hit the token limit. Problem: {problem_id}, idx: {j}")
                warning = WarningType.MINOR # model just didnt have time, any error could have been caused by this
            elif not is_correct and check_all_numbers(messages_problem[j][-1]["content"], str(problem["answer"])):
                logger.warning(f"Model answer: {model_answer} is not equal to gold answer: {gold_answer} even though model output contains the gold answer. Problem: {problem_id}, idx: {j}")
                warning = max(warning, WarningType.POSSIBLE)
            elif not is_correct and check_number_proximity_any_order(str(problem["answer"]), messages_problem[j][-1]["content"]):
                logger.warning(f"Numbers appearing in gold answer appear close together in model answer, but answer was incorrect. Problem: {problem_id}, idx: {j}")
                warning = max(warning, WarningType.POSSIBLE)
            elif len(messages_problem[j][-1]["content"]) == 0:
                logger.warning(f"Empty message in problem: {problem_id}, idx: {j}")
                warning = WarningType.MAJOR
            answers.append(model_answer)
            warnings.append(warning.value)
            corrects.append(is_correct)
        else:
            answers.append(None)
            warnings.append(0)
            corrects.append(0)
    try:
        if log_this:
            logger.info(f"Finished problem: {problem_id}, answers: {answers}, gold answer: {str(problem['answer'])}, #Correct: {sum(corrects)}")
    except:
        pass

    if final_answer:
        pass_at_1 = sum(x == gold_answer for x in answers)/n
    else:
        pass_at_1 = 0
    for i in range(len(costs_problem)):
        costs_problem[i]["cost"] = model_config["read_cost"] * costs_problem[i]["input_tokens"] + model_config["write_cost"] * costs_problem[i]["output_tokens"]
        costs_problem[i]["cost"] /= 10 ** 6
    cost = {
        "cost": sum([d["cost"] for d in costs_problem]),
        "input_tokens": sum([d["input_tokens"] for d in costs_problem]),
        "output_tokens": sum([d["output_tokens"] for d in costs_problem]),
    }

    with open(output_file, "w") as f:
        json.dump({
                    "idx": problem_idx,
                    "problem": problem_statement,
                    "gold_answer": str(problem.get("answer", "None")),
                    "source": problem.get("source", "None"),
                    "types": problem.get("problem_type", "None"),
                    "messages": messages_problem, 
                    "history": histories_problem,
                    "answers": [convert_answer(answer) for answer in answers],
                    "correct": corrects,
                    "pass_at_1": pass_at_1,
                    "cost": cost,
                    "detailed_costs": costs_problem,
                    "warnings": warnings,
                }, f, indent=2)

def convert_answer(answer):
    """Converts an answer to a string.

    Args:
        answer: The answer to convert.

    Returns:
        str: The string representation of the answer.
    """
    try:
        if type(answer) == sympy.Integer:
            return safe_str_int(int(answer))
        else:
            return safe_str_int(answer)
    except:
        return "None"
