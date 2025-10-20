"""
Dirty global object for debug request logging.
"""

import json
import os
from loguru import logger
from collections import OrderedDict


class RequestLogger:
    def __init__(self):
        self.log_dir = "logs/requests"

    def set_metadata(self, comp_name, solver_name, batch_idx_to_problem_idx):
        self.comp_name = comp_name
        self.solver_name = solver_name
        self.batch_idx_to_problem_idx = batch_idx_to_problem_idx

    def log_request(self, ts, batch_idx, request, **info):
        problem_idx = self.batch_idx_to_problem_idx[batch_idx]
        logfile = f"{self.log_dir}/{self.comp_name}/{self.solver_name}/{ts}_p{problem_idx}_idx{batch_idx}.json"
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        if os.path.exists(logfile):
            logger.warning(f"Can't log request, log file already exists: {logfile}")
            return

        data = OrderedDict(
            {
                "comp_name": self.comp_name,
                "solver_name": self.solver_name,
                "timestamp": ts,
                "problem_idx": problem_idx,
                "batch_idx": batch_idx,
                "request_info": info,
                "request": request,
            }
        )

        with open(logfile, "w") as f:
            json.dump(data, f, indent=4)

    def log_response(self, ts, batch_idx, response, **info):
        problem_idx = self.batch_idx_to_problem_idx[batch_idx]
        logfile = f"{self.log_dir}/{self.comp_name}/{self.solver_name}/{ts}_p{problem_idx}_idx{batch_idx}.json"
        if not os.path.exists(logfile):
            logger.warning(f"Can't log response, log file does not exist: {logfile}")
            return

        with open(logfile, "r") as f:
            data = json.load(f, object_pairs_hook=OrderedDict)

        # Update the data with the response information
        data["response_info"] = info
        data["response"] = response
        with open(logfile, "w") as f:
            json.dump(data, f, indent=4)


request_logger = RequestLogger()
