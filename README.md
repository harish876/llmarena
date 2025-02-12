# MathArena: Evaluating LLMs on Uncontaminated Math Competitions

MathArena is a platform for the evaluation of LLMs on the latest math competitions and olympiads. It is hosted on [matharena.ai](https://matharena.ai/). This repository contains all the code used for model evaluation of the competitions. The README explains how to run your models or add a new competition. 

## Installation

All required packages for installation are handled using [UV](https://github.com/astral-sh/uv). To install UV, run

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or
```bash
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Evaluating a New Model

To evaluate a new model on an existing competition, you will have to perform several (small) steps.

### Creating Config

Add a config of the model in the [`configs/`](configs/) folder. Each model config has to contain the following required arguments:
* `model`: The model name matching the API you want to use. Reasoning effort of OpenAI models can be set by appending `--[low/medium/high]` to the model name, e.g., `o3-mini--high`.
* `api`: The API to use for model evaluation. Currently, we support the following APIs, mentioned alongside the environment variable that needs to point to your API key:
    * [openai](https://openai.com/api/): OPENAI_API_KEY 
    * [anthropic](https://www.anthropic.com/api): ANTHROPIC_API_KEY
    * [together](https://www.together.ai/): TOGETHER_API_KEY
    * [google](https://ai.google.dev/): GOOGLE_API_KEY
    * [deepseek](https://api-docs.deepseek.com/): DEEPSEEK_API_KEY
    * [openrouter](https://openrouter.ai/): OPENROUTER_API_KEY
    * [vllm](https://docs.vllm.ai/en/latest/): no API key required, runs model locally.
* `human_readable_id`: A string describing the config in a human-readable manner. Must be unique among all configs.

Additionally, you can set a few optional parameters:
* Standard parameters: Any parameters your provider accepts in their API, such as `temperature`, `top_p`, and `top_k`. By default `temperature` is set to the default temperature of the competition (see [Adding a Competition](#adding-a-competition)) and the others are not set.
* `max_tokens`: Max number of tokens for the model. The parameter name is automatically adjusted for the different APIs. Defaults to the default max tokens for the competition  (see [Adding a Competition](#adding-a-competition)).
* `concurrent_requests`: Number of concurrent requests to the API provider. Defaults to 30.
* `timeout`: The timeout for the API provider. Defaults to 500.
* `max_retries`: Number of times to retry the request if it fails. Defaults to 50.
* `read_cost`: Cost of one million input tokens of the model in USD. Defaults to 1.
* `write_cost`: Cost of one million output tokens of the model in USD. Defaults to 1.

### Running the model
You can now run the model on any existing competition by executing the following command:
```bash
uv run python scripts/run.py --configs path/to/your/config --comp path/to/competition
```
Here, `path/to/your/config` is the path to your config relative to the `configs` folder and `path/to/competition` is the path to the competition relative to the `data` folder. For instance, to run `gpt-4o`on AIME 2025 I, you can run:
```bash
uv run python scripts/run.py --configs openai/gpt-4o.yaml --comp aime/aime_2025_I
```
This command accepts several optional parameters:
* `skip_existing`: If activated, skip existing runs in case you want to continue from a prior run.
* `n`: Number of times a model is run on each problem. Defaults to 4.

To run a model locally using vllm, you additionally need to start a server using:
```bash
vllm serve [[model_name]] --dtype auto --api-key token-abc123
```

The raw model results will be stored in `outputs/path/to/competition/[[model_name]]`. To see the results, you can run:
```bash
uv run python scripts/app.py --comp path/to/competition --models [[model_name]]
```
This will open a server on [http://localhost:5001/](http://localhost:5001/) that presents all results in a human-readable format. It is important to manually check the warnings that are visually presented in the app. The logs contain more information about the warnings, but any warning is caused by one of the following problems:
* The answer is empty, likely indicating an API error.
* The answer has $10^k \cdot 2^n$ output tokens, likely indicating that the model reached the maximum number of tokens.
* The correct answer is present in the model answer, but it was not extracted.
If you notice that a problem occurred, you can do one of the following:
* Delete the file associated with the problem and rerun the model with the `skip_existing` option. This will rerun the problem for that model.
* Fix the code (in case of a parsing error) and rerun the model with the `skip_existing` option. This will analyze the existing model results with the fixed parser.


## Adding a Competition
Adding a competition can be done in several quick steps. 

### Adding the Problems
In the `data` folder, add a new subfolder containing all information regarding your competition (name it appropriately). There are three main parts when you add a new competition:
- **Adding the problems**: In your competition folder, add a subfolder named `problems`. In this subfolder, create every problem as a separate LaTeX file. The file names should be `1.tex`, `2.tex`, ..., `{k}.tex`, where `k` is the number of problems in your competition. You can skip a problem if you want/need to.
- **Adding the answers**: Add a file `answers.csv` that contains two columns: `id` and `answer`. `id` should point to the filename of your problem (without `.tex`). The `answer` should be the answer to your problem. Currently, we only support integer answers. You can skip a problem if you want/need to.
- **Adding a config**: Add a file `config.yaml` that contains four parameters. For most purposes, you can copy-paste it from other competitions. 
    - `instruction`: Contains the instructions for a model. They are prepended to each problem statement. Make sure it contains an instruction to put the final answer in `\boxed` as this is what our parser is looking for. 
    - `default_temperature`: Contains the default temperature for each model if they are not specified in model-specific configs.
    - `default_max_tokens`: Contains the default max tokens for each model if they are not specified in model-specific configs.
    - `strict_parsing`: Whether or not to perform strict parsing. If strict, then an answer will only be counted as correct if it appears in the correct format (e.g., \boxed{43}). If not strict, the parser will do a fail-safe by extracting the last integer in the answer.
    - `n_problems`: The number of problems in the competition.

### Verifying Problem Statements
You should verify the correctness of the problem statements by running
```bash
uv run python scripts/check_latex.py --comp path/to/competition
```
Here `path/to/competition` is the path to your competition relative to the `data` folder. You should then build `latex/main.tex` and make sure the pdf contains no errors and shows each problem accurately.

### Running Models
To run models, you can either follow the instructions from [the previous section](#evaluating-a-new-model) or you can run a shortcut to run multiple models at the same time:
```bash
uv run python scripts/run_multiple.py --apis openai google anthropic together --comp path/to/competition
```
This will run all models in `configs/` that use any of the APIs specified. By default, models from different APIs are run in parallel. If you want to run all models in parallel (even the ones that use the same API), add `--simul`. Additionally, you can specify the following parameters:
* `models`: Space-separated regexes of model names. The script will only run a model if it matches any of the regexes.
* `skip_existing`: If activated, skip existing runs in case you want to continue from a prior run.
* `n`: Number of times a model is run on each problem. Defaults to 4.

Note that you should specify the appropriate API keys and run a VLLM server as in [the previous section](#evaluating-a-new-model). Logs for the runs will be visible in the `logs` folder.

### Viewing Results
To view the results for your competition, run 
```bash
uv run python scripts/app.py --comp path/to/competition
```
This will open a server on [http://localhost:5001/](http://localhost:5001/) that presents all results in a human-readable format.