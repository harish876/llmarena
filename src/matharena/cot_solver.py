"""This module defines a Chain-of-Thought (CoT) solver for math problems."""

from loguru import logger

class CoTSolver:
    """
    A solver that uses a simple Chain-of-Thought prompting approach.
    """
    def __init__(self, querier):
        """
        Initializes the CoTSolver.

        Args:
            querier: An object that can be used to query a language model.
        """
        self.querier = querier
    
    def solve(self, problems, **tokenizer_kwargs):
        """
        Solves a list of problems using the CoT approach.

        Args:
            problems (list): A list of problem statements to be solved.
            **tokenizer_kwargs: Additional keyword arguments for the tokenizer.

        Yields:
            tuple: A tuple containing the index, solution, detailed cost, and history for each problem.
        """
        queries = [
            [{
                "role": "user",
                "content": problem[0] if isinstance(problem, tuple) else problem
            }] for problem in problems
        ]
        for idx, response, detailed_cost in self.querier.run_queries(queries, **tokenizer_kwargs):
            yield idx, response, detailed_cost, None