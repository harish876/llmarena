"""This module defines the base Agent class for solving math problems."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, override

from loguru import logger
from tqdm import tqdm

from matharena.api_client import APIClient
from matharena.solvers import BaseSolver, SelfcheckAgent, SolverResponse


class AgentPool(BaseSolver):
    """
    A solver that manages a pool of agents to solve problems.
    """

    AGENT_CLASSES = {"selfcheck": SelfcheckAgent}

    def __init__(self, solver_config, default_prompt_template, default_api_client_args, last_chance_prompt):
        super().__init__(solver_config, default_prompt_template, default_api_client_args, last_chance_prompt)

        # AgentPool handles multithreading, individual agents use
        # model_config["concurrent_requests"] internally, but usually send 1 query at a time to APIClient.
        self.scaffold_config = self.solver_config["scaffold_config"]
        self.n_threads = self.scaffold_config.get("n_threads", 1)
        self.AGENT_CLASS = AgentPool.AGENT_CLASSES[self.scaffold_config["scaffold_name"]]

    def _run_agent(self, batch_idx: int, stmt: tuple[str, Any]):
        agent = self.AGENT_CLASS(
            batch_idx=batch_idx,
            solver_config=self.solver_config,
            default_prompt_template=self.default_prompt_template,
            default_api_client_args=self.default_api_client_args,
        )
        # TODO: implement image support for agents if needed
        return agent.solve(stmt[0])

    @override
    def solve_batch(self, stmt_batch: list[tuple[str, Any]]):
        """
        Solves a batch of problems. Handles multithreading, launching one Agent per problem.

        Args:
            stmt_batch (list[tuple[str, Any]]): A batch of problem statements as (text, image) pairs.

        Yields:
            solver_response: A SolverResponse object containing the batch_index, the conversation array, detailed cost, and history for each problem.
        """
        logger.info(f"Starting agents with n_threads={self.n_threads} for batch of size {len(stmt_batch)}.")
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = []
            for batch_idx, stmt in enumerate(stmt_batch):
                futures.append(executor.submit(self._run_agent, batch_idx, stmt))
            iterator = as_completed(futures)
            iterator = tqdm(iterator, total=len(futures))
            for future in iterator:
                solver_response = future.result()
                yield solver_response

    @override
    def last_chance(self, previous_response: SolverResponse) -> SolverResponse:
        """
        If the parser did not find the solution for some problem, the solver has a last chance to modify its response.
        TODO: see if we want last chance for agents, for now it's skipped as agents have plenty of chances to
        internally fix their mistakes and make sure they provided an answer.

        Args:
            previous_response (SolverResponse): The response this solver previously returned for a problem.

        Returns:
            SolverResponse: The modified response after reprompting the model to report an answer.
        """
        last_chance_query = [{"role": "user", "content": self.last_chance_prompt}]
        empty_response = [{"role": "assistant", "content": ""}]
        logger.info(
            "AgentPool.last_chance called, but currently not implemented for agents. "
            "Returning previous response with empty assistant message appended."
        )
        return SolverResponse(
            idx=0,
            conversation=previous_response.conversation + last_chance_query + empty_response,
            detailed_cost=previous_response.detailed_cost,
            history=previous_response.history,
        )
