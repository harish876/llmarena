from typing import Any

from loguru import logger
from matharena.solvers import SolverResponse
from matharena.api_client import APIClient
import time
import threading


class BaseAgent:
    """
    An abstract agent that solves a single math problem instance by using one or more APIClients.

    batch_idx: the index of this problem in the batch handled by the AgentPool.
    solver_config: the full solver config, including model_config and scaffold_config.
    default_prompt_template: the prompt template
        (the "instruction" from the competition config + {problem} template)
    default_api_client_args: the kwargs for the APIClient constructor, i.e., all kwargs
        stated in model_config + "tools" and "max_tool_calls" from the competition config.
        Tools is a list of pairs (function, tool_spec) where function is None for responses API
        The agent can override these when creating its own APIClient(s).
    """

    def __init__(self, batch_idx, solver_config, default_prompt_template, default_api_client_args):
        self.batch_idx = batch_idx
        self.bi = batch_idx  # short alias
        self.solver_config = solver_config
        self.default_prompt_template = default_prompt_template
        self.default_api_client_args = default_api_client_args
        self._lock = threading.Lock()

    def _start_run(self, stmt: str):
        """
        Starts the run:
            - Resets the state to be returned.
            - Starts the timer.
            - Sets the 1st of 2 entries in the final conversation.
        """
        self.stmt = stmt
        logger.debug(f"[{self.bi}] Starting agent run for problem: {stmt[:50]}...")
        self.conversation = [{"role": "user", "content": stmt}, {"role": "assistant", "content": "TODO"}]
        self.detailed_cost = {"cost": 0, "input_tokens": 0, "output_tokens": 0, "time": "TODO"}
        self.history = []
        self.start_time = time.time()

    def _query(self, client: APIClient, query: list[dict[str, Any]]):
        """
        A wrapper that runs a single query (conversation) via the given APIClient and updates the cost state.
        Queries should add user/developer messages in clean format or reuse message blocks from same client.
        """
        ret = list(client.run_queries([query], no_tqdm=True, custom_indices=[self.batch_idx]))
        _, conversation, detailed_cost = ret[0]
        with self._lock:
            self.detailed_cost["cost"] += detailed_cost["cost"]
            self.detailed_cost["input_tokens"] += detailed_cost["input_tokens"]
            self.detailed_cost["output_tokens"] += detailed_cost["output_tokens"]
        return conversation

    def _add_history(self, step: str, timestep: int, conversation: Any, **kwargs):
        """
        Adds an entry to the history.
        """
        entry = {
            "step": step,
            "timestep": timestep,
            "messages": conversation,
        }
        entry.update(kwargs)
        with self._lock:
            self.history.append(entry)

    def _end_run(self, final_response: str) -> SolverResponse:
        """
        Ends the run:
            - Stops the timer and updates the detailed_cost time.
            - Sets the 2nd of 2 entries in the final conversation.
            - Returns a SolverResponse object with batch index.
        """
        logger.debug(f"[{self.bi}] Ending agent run for problem: {self.stmt[:50]}...")
        end_time = time.time()
        self.detailed_cost["time"] = end_time - self.start_time
        self.conversation[1]["content"] = final_response
        return SolverResponse(
            idx=self.batch_idx, conversation=self.conversation, detailed_cost=self.detailed_cost, history=self.history
        )

    def solve(self, stmt: str) -> SolverResponse:
        """
        Solves a single problem statement.

        Args:
            stmt (str): A problem statement as text.

        Returns:
            SolverResponse: A SolverResponse object containing:
             - index: set to 0 here
             - conversation: the conversation array, for agents it should just have 2 blocks: "user" and "assistant"
             - detailed_cost: agent must report detailed cost info: cost, in/out tokens, time
             - history: a list of steps, where each step corresponds to one conversation:
                 - "step": unique string id
                 - "timestep": the time at which this step happened (for visualization)
                 - "messages": the full conversation in this step
                 - any extra debug keys
                 The agent decides what to report as the final response in the conversation.
        """
        raise NotImplementedError("Subclasses should implement this method.")
