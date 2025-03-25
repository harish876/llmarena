# MathArena: Evaluating LLMs on Uncontaminated Math Competitions

MathArena is a platform for the evaluation of LLMs on the latest math competitions and olympiads. It is hosted on [matharena.ai](https://matharena.ai/). This repository contains all the code used for model evaluation of the competitions. The README explains how to run your models or add a new competition. 

## Table of Contents
- [Installation](#installation)
- [Evaluating a New Model](#evaluating-a-new-model)
  - [Model Configuration](#model-configuration)
  - [Running the Model](#running-the-model)
  - [Local VLLM Usage](#running-models-locally-using-vllm)
- [Adding a Competition](#adding-a-competition)
  - [Setting Up Competition Files](#setting-up-competition-files)
  - [Verifying Problem Statements](#verifying-problem-statements)
  - [Running Models on Competitions](#running-models-on-competitions)
  - [Competitions Requiring Grading](#competitions-requiring-grading)
  - [Running LLMs as Judges](#running-llms-as-judges)
- [Viewing Results](#viewing-results)
- [Evaluation logs](#evaluation-logs)

## Installation

MathArena uses [UV](https://github.com/astral-sh/uv) to manage dependencies. If you want to run local models, uncomment the vllm installation in `pyproject.toml`.

### Install UV

- **macOS and Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Windows:**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
---

### Alternative installation

As an alternative to UV, you can also create a conda environment and install the package as follows:
```bash
conda create -n matharena python=3.12
conda activate matharena
python -m pip install -e .
```
If you choose this option, disregard `uv run` in all instructions and use python directly instead.

## Evaluating a New Model

### Model Configuration

Create a configuration file in the `configs/` folder. Each config must include:
- **Required:**
  - `model`: Model name. Reasoning effort of OpenAI models can be set by appending `--[low/medium/high]` to the model name, e.g., `o3-mini--high`.
  - `api`: API provider. The API key should be defined as environment variable when using the specified API. The supported options with their corresponding API keys are:
    - **openai**: `OPENAI_API_KEY`
    - **anthropic**: `ANTHROPIC_API_KEY`
    - **together**: `TOGETHER_API_KEY`
    - **google**: `GOOGLE_API_KEY`
    - **deepseek**: `DEEPSEEK_API_KEY`
    - **openrouter**: `OPENROUTER_API_KEY`
    - **vllm**: (runs locally; no API key required)
  - `human_readable_id`: A unique, descriptive identifier.
- **Optional Parameters:**
  - API settings like `temperature`, `top_p`, and `top_k` (default: `temperature` is from competition config, see [Adding a Competition](#adding-a-competition)).
  - `max_tokens`: Max number of tokens for the model (default: from competition config, see [Adding a Competition](#adding-a-competition)).
  - `concurrent_requests`: Number of parallel requests to API (default: 30).
  - `timeout`: Request timeout in seconds (default: 500).
  - `max_retries`: Retry attempts to API (default: 50).
  - `read_cost` & `write_cost`: Cost per million tokens in USD for input and output tokens (default: 1 each).
  - `date`: Creation date of the model in the format "yyyy-mm-dd". Only necessary to ensure automatic inclusion of the asterisk on the website for competitions that happen before this date.
  - `batch_processing`: If set to True, the model will be queried using batch processing. This mode is only available for OpenAI and Anthropic models and cuts the cost of the model in half, but it will increase the time required to obtain the completions.

### Running the model
Execute the following command to evaluate a model on a competition:
```bash
uv run python scripts/run.py --configs path/to/your/config --comp path/to/competition
```
- `path/to/your/config`: Relative path from the `configs/` folder to the model configuration.
- `path/to/competition`: Relative path from the `data/` folder to the competition folder.

**Example:**
```bash
uv run python scripts/run.py --configs openai/gpt-4o.yaml --comp aime/aime_2025_I
```

**Additional Flags:**
- `skip_existing`: Skip problems already processed through the model.
- `n`: Number of runs per problem (default: 4).

### Running Models Locally Using VLLM

If using a local model with vllm, start the server:
```bash
vllm serve [[model_name]] --dtype auto --api-key token-abc123
```

## Adding a Competition
Adding a competition can be done in several quick steps. 

### Setting Up Competition Files
In the `data/` folder, create a new directory for your competition with the following structure:
1. **Problems:**  
   - Create a subfolder `problems/` and add each problem as a separate LaTeX file named `1.tex`, `2.tex`, …, `{k}.tex`, where `k` is the number of problems in your competition. You can skip a problem if you want/need to.
2. **Competition Config:**  
   - Create a `config.yaml` with:
     - `instruction`: Instructions for the model. *Must* require the final answer be in `\boxed{}` (for correct parsing).
     - `default_temperature`: Default temperature for runs.
     - `default_max_tokens`: Default max tokens.
     - `strict_parsing`: `true` for strict format matching (e.g., only `\boxed{43}` is accepted) or `false` for lenient parsing.
     - `n_problems`: Total number of problems.
     - `date`: Date of the competition, in the format "YYYY-MM-DD".
     - `final_answer` (optional): If set to false, the competition is one that is manually graded with judges. Defaults to true if not set.
3. **Answers:**  
   - If the competition is one based on final answers, add an `answers.csv` file with columns `id` and `answer`.
     - `id`: The problem filename (without the `.tex` extension).
     - `answer`: The integer answer.
   - If the competition is evaluated using human judges, add a `grading_scheme.json` file. This file should consist of a list of dictionaries, each of which contain the following fields:
     - `id`: The problem filename (without the `.tex` extension).
     - `points`: The maximum number of points to be gained for the question.
     - `scheme`: A list of dictionaries, each containing specific substeps for which points can be given. Specifically, each dictionary contains the following keys:
        - `points`: Points associated with this step.
        - `title`: Title of the step. Should be unique across all dictionaries in this scheme.
        - `desc`: Description of the step.

### Verifying Problem Statements
Ensure your LaTeX problems compile correctly:
```bash
uv run python scripts/check_latex.py --comp path/to/competition
```
Then, build the `latex/main.tex` to generate a PDF and confirm all problems appear as expected.

### Running Models on Competitions
To run multiple models (possibly across different APIs), use:
```bash
uv run python scripts/run_multiple.py --apis openai google anthropic together --comp path/to/competition
```
This will run models from the same API sequentially and from different APIs concurrently.
**Options:**
- `--simul`: Run all models in parallel, even if they use the same API.
- `models`: Provide space-separated regex patterns to filter models. A model is only run if it matches any of the regexes.
- `skip_existing`: Skip problems already processed through the model.
- `n`: Number of runs per problem (default: 4).

*Note:* For local vllm usage, ensure the vllm server is running as described above. Logs will be found in the `logs/` folder.

### Competitions Requiring Grading
To set up grading of questions, convert the model answers to TeX files:
```bash
uv run python scripts/judge/answers_to_latex.py --comp path/to/competition
```
This will compile all model answers in a PDF file in the folder `latex/path/to/competition`. Double-check the produced PDFs to check for compilation errors. They should be manually fixed.

Now, collect all PDFs for all evaluated models in a single folder using:
```bash
uv run python scripts/judge/collect_pdfs.py --comp path/to/competition
```
This will put all PDFs associated with question with idx `i` in the folder `latex/path/to/competition/i`. Each PDF will be given a unique (anonymous) ID. Follow the instructions in `README_judges.md` to grade each PDF. The grading of a single PDF should be placed in `grading/path/to/competition/i/{ID}.json`, where the ID is the ID given to the PDF associated with the grading. In case a PDF is graded by multiple people, you can add more files by naming them `grading/path/to/competition/i/{ID}_{X}.json` where `X` is any suffix. Finally, run
```bash
uv run python scripts/judge/merge_judgments.py --comp path/to/competition
```
This will add the judgments directly in the raw output traces of the models.

### Running LLMs as judges
To run an LLM as a judge, you must first add the solutions of all problems of the competition in `data/path/to/competition/solutions/{i}.text` where `i` is the index of the problem. 

Then, use the following command:
```bash
uv run python scripts/grade.py --grader_config path/to/grader --solver_config path/to/solver/1 path/to/solver/2 --comp path/to/competition
```

**Options:**
- `path/to/grader`: Relative path from the `configs/` folder to the model configuration for the judge.
- `path/to/solver`: Relative path from the `configs/` folder to the model configuration of the judged model. Multiple ones can be given by passing space-separated paths.
- `path/to/competition`: Relative path from the `data/` folder to the competition folder.

**Example:**
```bash
uv run python scripts/grade.py --grader_config openai/o3-mini.yaml --solver_config openai/o3-mini.yaml anthropic/claude-37.yaml --comp usamo/usamo_2025
```

**Additional Flags:**
- `skip_existing`: Skip problems already processed through the model.
- `n`: Number of runs per problem to evaluate (default: 4). Must be no larger than the amount of generated solutions.

*Notes:* For local vllm usage, ensure the vllm server is running as described above. Logs will be found in the `logs/` folder. We also recommend either using generalist models or either of `o1`, `o3-mini` or `Claude 3.7` as graders due to their robustness with respect to the formatting instructions. 

After obtaining the judgments, you can then process them using the aforementioned `merge_judgments.py` script:
```bash
uv run python scripts/judge/merge_judgments.py --comp path/to/competition --grading-folder autogradings
```
This will add the judgments directly in the raw output traces of the models.

## Viewing Results

Launch a local web server to inspect the results:
```bash
uv run python scripts/app.py --comp path/to/competition
```
Access the app at [http://localhost:5001/](http://localhost:5001/). Warning signs for solutions indicate a potential problem with the model run and should be manually verified. Any warning is caused by one of the following problems:

* 💀: parser threw an error or encountered something unexpected.
* ⚠️: The correct answer might be present in the model answer, but it was not extracted.
* ❕: Model likely hit max token limit.

If issues are found, delete the corresponding output file or fix the parser and rerun the model with `skip_existing`. If the parser requires a manual overwrite, you can edit `src/matharena/parse_manual.py` and add a key-value pair mapping the model solution to a parseable solution.

## Evaluation Logs

You can find logs from our evaluation containing full reasoning traces (if available) and solutions produced by the models at the following link: [https://files.sri.inf.ethz.ch/matharena/matharena_data.zip](https://files.sri.inf.ethz.ch/matharena/matharena_data.zip).

## Citation

```
@misc{balunovic_srimatharena_2025,
	title = {MathArena: Evaluating LLMs on Uncontaminated Math Competitions},
  author = {Mislav Balunović and Jasper Dekoninck and Ivo Petrov and Nikola Jovanović and Martin Vechev},
	copyright = {MIT},
	url = {https://matharena.ai/},
	publisher = {SRI Lab, ETH Zurich},
	month = feb,
	year = {2025},
}
```
