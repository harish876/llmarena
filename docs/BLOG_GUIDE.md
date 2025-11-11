# Blog Guide: Analyzing Model Performance from MathArena Outputs

## Task at hand
 - Coming up with ideas/scripts to easily analyze data in our benchmark experiments.
 - Any tool can be used, we need this to be interactive so analysis data has to be in the form of JSON and not a matplotlib chart.
 - For starters we need to come up with ideas on what exactly can be written in the blog.
 - My ideas are, for every competition
    -  Question by Question breakdown - answers by the LLM and logs
    -  Top Model performance by accuracy, cost, latency
    -  Patterns observed in the data - this would be tricky but what type of questions does the LLM get right, does it get wrong, if yes then why insights. Can we find any distinction between a good performing model and a bad one?

## Overview

This guide explains the structure of MathArena evaluation outputs and outlines the types of analyses you can perform for writing a blog post about model performance. All outputs are stored in AWS S3 bucket: `pilotcrew-dev-autoeval`.

## Output Structure

### Directory Hierarchy

The outputs follow this structure:
```
outputs/
├── {competition_category}/
│   ├── {competition_name}/
│   │   ├── {api_provider}/          # e.g., "openrouter", "bedrock"
│   │   │   ├── {model_name}/        # e.g., "gpt-5", "claude-4-opus"
│   │   │   │   ├── 0.json          # Problem 0 results
│   │   │   │   ├── 1.json          # Problem 1 results
│   │   │   │   ├── 2.json          # Problem 2 results
│   │   │   │   └── ...
│   │   ├── leaderboard_summary.json # Aggregated leaderboard
│   │   ├── model_list.json          # List of models evaluated
│   │   └── rank_variance.jsonl     # Ranking variance analysis
```

**Example path:**
- `outputs/aime/aime_2025/openrouter/gpt-5/0.json` - GPT-5's results on problem 0 of AIME 2025

### Individual Problem Output Files (`.json`)

Each problem output file (e.g., `0.json`) contains:

#### Problem Information
- `idx`: Problem index/number
- `problem`: The problem statement text
- `gold_answer`: The correct answer
- `source`: Source of the problem (if available)
- `types`: Problem type/category tags (if available)

#### Performance Metrics
- `N`: Number of runs completed for this problem (typically 4)
- `pass_at_1`: Pass@1 score (fraction of runs that were correct on first attempt)
- `answers`: Array of answers given by the model across all runs
- `correct`: Array of boolean values indicating if each run was correct
- `warnings`: Array of warning counts per run

#### Cost and Resource Usage
- `cost`: Aggregate cost object containing:
  - `cost`: Total cost in dollars
  - `input_tokens`: Total input tokens used
  - `output_tokens`: Total output tokens generated
  - `time`: Total time taken (in seconds)
- `detailed_costs`: Array of cost breakdowns for each individual run:
  - `cost`: Cost for this run
  - `input_tokens`: Input tokens for this run
  - `output_tokens`: Output tokens for this run
  - `time`: Time taken for this run

#### Raw Data
- `messages`: Array of conversation arrays (one per run), each containing:
  - User message (the problem prompt)
  - Assistant messages (model's reasoning and response)
  - Message types (e.g., "cot" for chain-of-thought, "response" for final answer)
- `judgment`: Grading/judgment details for each run
- `history`: Step-by-step history of agent actions (if using agent-based solver)

### Aggregated Files

#### `leaderboard_summary.json`
Contains aggregated performance across all problems for each model:
- `model_display_name`: Human-readable model name
- `config_path`: Model configuration path
- `avg_score`: Average accuracy score (0-1 scale)
- `avg_cost`: Average cost per problem
- `rank`: Ranking range (if variance analysis was done)
- `date`: Model evaluation date
- `config`: Full model configuration details
- `per_competition_scores`: Scores broken down by competition

#### `rank_variance.jsonl`
Contains ranking variance analysis results (if computed), showing confidence intervals for model rankings.

## Analytics Opportunities

### 1. Cost Analytics

#### Per-Problem Cost Analysis
- **Cost distribution**: Analyze how costs vary across different problems
- **Cost outliers**: Identify problems that are particularly expensive (high token usage)
- **Cost vs. difficulty**: Correlate problem cost with problem difficulty or type
- **Token efficiency**: Compare input vs. output token ratios across models
- **Cost per correct answer**: Calculate cost efficiency (cost divided by accuracy)

#### Model-Level Cost Comparison
- **Average cost per problem**: Compare total cost across models
- **Cost per accuracy point**: Cost-effectiveness metric (cost / accuracy)
- **Cost trends over time**: Track how model costs change with different versions
- **Provider comparison**: Compare costs between different API providers (OpenRouter vs. Bedrock)

#### Cost Breakdown Analysis
- **Input vs. output costs**: Analyze which component drives costs more
- **Reasoning overhead**: For models with chain-of-thought, analyze cost of reasoning steps
- **Token usage patterns**: Identify if certain problem types require more tokens

### 2. Latency/Performance Analytics

#### Response Time Analysis
- **Average latency per problem**: Compare how long models take to respond
- **Latency distribution**: Identify problems that take unusually long
- **Latency vs. accuracy trade-off**: Do faster models sacrifice accuracy?
- **Latency vs. cost**: Are faster models also cheaper?

#### Throughput Analysis
- **Problems per hour**: Calculate evaluation throughput
- **Concurrent request efficiency**: Analyze how well models handle parallel requests
- **Time per token**: Calculate latency efficiency metrics

#### Consistency Analysis
- **Latency variance**: How consistent is response time across runs?
- **Problem-specific latency**: Do certain problem types take longer?

### 3. Accuracy Analytics

#### Overall Performance
- **Average accuracy**: Overall correctness across all problems
- **Pass@1 scores**: First-attempt success rates
- **Consistency**: How often does the model get the same answer across multiple runs?
- **Confidence intervals**: Statistical significance of accuracy differences

#### Problem-Type Analysis
- **Accuracy by problem type**: Performance breakdown by category (e.g., algebra, geometry)
- **Difficulty analysis**: How do models perform on easy vs. hard problems?
- **Error patterns**: What types of mistakes do models make?

#### Comparative Analysis
- **Model rankings**: Which models perform best overall?
- **Head-to-head comparisons**: Direct model comparisons on specific problems
- **Performance gaps**: Identify where models struggle most
- **Improvement over time**: Track accuracy improvements across model versions

#### Competition-Specific Analysis
- **Per-competition scores**: How do models perform on different benchmarks (AIME, AGIEval, etc.)?
- **Competition difficulty comparison**: Which competitions are hardest?
- **Model specialization**: Do certain models excel at specific competition types?

### 4. Cross-Metric Analysis

#### Cost-Performance Trade-offs
- **Cost vs. accuracy curves**: Visualize the efficiency frontier
- **Best value models**: Identify models with best accuracy-to-cost ratio
- **Premium vs. budget models**: Compare high-cost high-accuracy vs. low-cost models

#### Latency-Performance Trade-offs
- **Speed vs. accuracy**: Do faster models sacrifice correctness?
- **Optimal model selection**: Find models that balance speed and accuracy

#### Multi-Dimensional Analysis
- **Cost-latency-accuracy 3D analysis**: Three-way trade-off visualization
- **Model efficiency scores**: Composite metrics combining all factors
- **ROI analysis**: Return on investment for different model choices

### 5. Trend Analysis

#### Temporal Trends
- **Model evolution**: How do newer model versions compare to older ones?
- **Performance improvements**: Track accuracy improvements over time
- **Cost reductions**: Monitor if newer models are more cost-efficient

#### Problem-Specific Trends
- **Problem difficulty trends**: Which problems are consistently hard?
- **Model consistency**: Do models perform consistently across similar problems?
- **Error clustering**: Identify patterns in where models fail

### 6. Advanced Analytics

#### Statistical Analysis
- **Confidence intervals**: Statistical significance of performance differences
- **Ranking variance**: How stable are model rankings?
- **Significance testing**: Which performance differences are statistically meaningful?

#### Comparative Benchmarks
- **Human performance comparison**: How do models compare to human scores?
- **Baseline comparisons**: Compare against simple baselines or previous SOTA
- **Ensemble potential**: Could combining models improve performance?

#### Failure Analysis
- **Error categorization**: Classify types of errors (calculation, reasoning, parsing)
- **Failure modes**: Identify common failure patterns
- **Recovery analysis**: Do models correct themselves in later runs?

## Data Access

### AWS S3 Location
All outputs are stored in: `s3://pilotcrew-dev-autoeval/outputs/`

The structure in S3 mirrors the local `outputs/` directory structure.

### Key Files to Access
1. **Individual problem results**: `{competition}/{competition_name}/{provider}/{model}/{problem_id}.json`
2. **Leaderboard summaries**: `{competition}/{competition_name}/leaderboard_summary.json`
3. **Model lists**: `{competition}/{competition_name}/model_list.json`

## Visualization Ideas

### Cost Analytics
- Cost distribution histograms
- Cost vs. accuracy scatter plots
- Cost breakdown pie charts (input vs. output tokens)
- Cost efficiency rankings

### Latency Analytics
- Latency distribution plots
- Latency vs. accuracy trade-off curves
- Response time heatmaps by problem type

### Accuracy Analytics
- Accuracy leaderboards
- Accuracy by problem type bar charts
- Confusion matrices for error analysis
- Performance improvement timelines

### Combined Visualizations
- Cost-accuracy efficiency frontiers (Pareto efficient models)
- Multi-dimensional radar charts
- Model comparison matrices
- Trade-off analysis dashboards
- Latency vs. accuracy scatter plots with interactive filtering

## Reference Examples

For inspiration on how to present benchmark results effectively, refer to these examples:

### Vals AI (https://www.vals.ai/home)
- **Pareto Efficient Models**: Shows cost vs. accuracy scatter plots highlighting models on the efficiency frontier
- **Industry-Specific Leaderboards**: Separate leaderboards for different domains (Legal, Coding, Finance, Healthcare, Math, Academic, Education)
- **Multiple Visualization Modes**: Interactive scatter plots, bar charts, and tables for the same data
- **Model Comparison Features**: Side-by-side comparisons of specific models
- **Update Feed**: Regular updates when new models are evaluated with key findings
- **Multi-dimensional Analysis**: Shows latency vs. accuracy, cost per test, and overall index scores

### Artificial Analysis (https://artificialanalysis.ai/evaluations/aime-2025)
- **Competition-Specific Evaluations**: Detailed breakdowns for specific competitions like AIME 2025
- **Model Rankings**: Clear leaderboard presentation with rankings and scores