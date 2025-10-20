import argparse
import csv
import json
import os

import yaml
from fasthtml.common import *
from pyparsing import srange
from torch import ScriptDict

from matharena.configs import extract_existing_configs

"""
    A dashboard app that shows all about a run 
"""

parser = argparse.ArgumentParser()
parser.add_argument("--comp", type=str)
parser.add_argument("--models", type=str, nargs="+", default=None)
parser.add_argument("--port", type=int, default=5001)
parser.add_argument("--output-folder", type=str, default="outputs")
parser.add_argument("--config-folder", type=str, default="configs/models")
parser.add_argument("--competition-config-folder", type=str, default="configs/competitions")
args = parser.parse_args()

current_comp = args.comp if args.comp is not None else "imo/imo_2025"  # fast-loading default comp

# Find all comps, directories below outputs of depth 2
all_comps = []
for root, dirs, files in os.walk(args.output_folder):
    rel_path = os.path.relpath(root, args.output_folder)
    depth = rel_path.count(os.sep)
    if depth == 1:
        all_comps.append(rel_path)
print(all_comps)
# sort by date modified
all_comps = sorted(all_comps, key=lambda x: os.path.getmtime(os.path.join(args.output_folder, x)), reverse=True)


def analyze_run(competition, models):
    configs, human_readable_ids = extract_existing_configs(
        competition,
        args.output_folder,
        args.config_folder,
        args.competition_config_folder,
        allow_non_existing_judgment=True,
    )
    if models is not None:
        for config_path in list(human_readable_ids.keys()):
            if human_readable_ids[config_path] not in models:
                del human_readable_ids[config_path]
                del configs[config_path]
    out_dir = os.path.join(args.output_folder, competition)

    results = {}
    for config_path in human_readable_ids:
        model_comp_dir = os.path.join(out_dir, config_path)
        results[f"{human_readable_ids[config_path]}"] = {}
        for problem_file in os.listdir(model_comp_dir):
            if not problem_file.endswith(".json"):
                continue
            problem_idx = int(problem_file.split(".")[0])
            with open(os.path.join(model_comp_dir, problem_file), "r") as f:
                data = json.load(f)
                results[f"{human_readable_ids[config_path]}"][problem_idx] = data
    return results


def load_sources(comp):
    sources = {}
    with open(f"{args.competition_config_folder}/{comp}.yaml", "r") as f:
        competition_config = yaml.safe_load(f)
    source_path = f"{competition_config['dataset_path']}/source.csv"
    if os.path.exists(source_path):
        with open(source_path, "r") as f:
            reader = csv.reader(f)
            problems = [row for row in reader][1:]
            sources = {int(row[0]): row[1] for row in problems}
    return sources


# Analyze run
results = analyze_run(current_comp, args.models)
boxes_expanded = False

# Get problem names
sources = load_sources(current_comp)

app, rt = fast_app(
    live=False,
    hdrs=[
        Meta(name="color-scheme", content="only light"),
        # KatexMarkdownJS(),
        Style(
            """
    :root {
        color-scheme: only light !important;
    }
    body, html {
        background-color: white !important;
        color: #333 !important;
    }
    h1, h2, h3, h4, h5, h6, summary, p {
        color: #333 !important;
    }
    * {
        color-scheme: only light !important;
    }
    .sidebar {
        display: inline-block;
        width: 30%;
        min-width: 30%;
        height: 100%;
        overflow-y: auto;
        background-color: #f8f9fa;
        padding: 20px;
        border-right: 1px solid #dee2e6;
        padding-right: 20px;
        z-index: 100;
    }
    .sidebar-list {
        max-height: 1000px;
        position: relative;
        overflow-y: scroll;
    }
    .expanded-1 {
        max-height: none !important;
        overflow-y: visible;
    }
    #comp-dropdown {
        padding: 0.5rem;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        background-color: white;
        font-size: 0.9rem;
        min-width: 200px;
        margin-bottom: 1rem;
    }
    .sidebar-item {
        display: block;
        padding: 8px;
        color: #333;
        text-decoration: none;
        border-radius: 4px;
    }
    .reload-button {
        font-style: italic;
        color: #002f94;
    }
    .sidebar-item.current {
        background-color: #e9ecef;
    }
    .sidebar-item:hover {
        background-color: #e9ecef;
    }
    .main {
        display: inline-block;
        width: 70%;
        overflow-x: auto;
        height: 100%;
        margin: 0% 2%;
    }
    .strong {
        font-weight: bold;
    }
    .fake-hr {
        border-bottom: 5px solid #333;
        margin: 1rem 0;
    }
    .problem-stats {
        white-space:pre;
        font-family:monospace;
    }
    .box {
        width: 100%;
        margin: 0rem 0rem 1.5rem 0rem;
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        max-height: 500px;
        position: relative;
        overflow-y: scroll;
    }
    .problem-box {
        background-color: #c7d9ff;
        white-space: pre-wrap;
        tab-size: 4;
    }
    .solution-box {
        background-color: #ffd700;
        border: 2px solid;
    }
    .response-box {
        background-color: #ffe4c8;
        white-space: pre-wrap;
        tab-size: 4;
        font-weight: normal;
    }
    .response-box-details {
        padding: 0rem 0rem;
    }
    .answer-box {
          white-space: pre-wrap;
          tab-size: 4;
    }
    .details-box {
        white-space: pre-wrap;
        tab-size: 4;
    }
    .correct {
        background-color: #c7ffcb;
    }
    .incorrect {
        background-color: #ffcbc7;
    }
    details > summary {
        list-style-type: '▶️ ';
    }
    details[open] > summary {
        list-style-type: '🔽 ';
    }
    details summary::after {
        display: none;
    }
    .user {
        background-color: #d1bca5;
    }
    .assistant {
        background-color: #ffe4c8;
    }
    .problem-image {
        width: 50%;
        display: block;
        margin: 2rem auto;
    }
    #history-step-selector {
        padding: 0.5rem;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        background-color: white;
        font-size: 0.9rem;
        min-width: 200px;
        margin-bottom: 1rem;
    }
    #history-step-selector:focus {
        outline: 2px solid #002f94;
        border-color: #002f94;
    }
    .conversation-content {
        border-left: 3px solid #dee2e6;
        padding-left: 1rem;
    }
    .error {
        color: #dc3545;
        padding: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .copy-button {
        font-size: 2rem;
    }
"""
        ),
        Script(
            """
document.addEventListener('DOMContentLoaded', function() {
    const dropdown = document.getElementById('comp-dropdown');
    if (dropdown) {
        dropdown.addEventListener('change', function() {
            const selectedComp = this.value.replace(/\//g, '---');
            // Redirect to /refresh/COMP/CURRENTURL
            window.location.href = `/refresh/${selectedComp}`;
        });
    }
});

function copyToClipboard() {
    // Get the parent element of the button
    const button = event.target;
    const parentDiv = button.closest('div');

    // Get all sibling P elements
    const siblings = Array.from(parentDiv.children).filter(el => el.tagName === 'P');

    // Extract innerHTML from each sibling, one per line
    const text = siblings.map(el => el.innerHTML).join('\\n');

    // Copy to clipboard
    navigator.clipboard.writeText(text).then(function() {
        // Visual feedback
        const originalText = button.textContent;
        button.textContent = '[Copied!]';
        setTimeout(function() {
            button.textContent = originalText;
        }, 1500);
    }).catch(function(err) {
        console.error('Failed to copy: ', err);
        alert('Failed to copy to clipboard');
    });
}
"""
        ),
    ],
)
title = f"MathArena App"


def get_problem_stats(results, model, problem):
    if type(problem) == str:
        problem = int(problem)
    res = results[model][problem]
    corrects = res["correct"]
    warnings = res.get("warnings", [False] * len(corrects))
    if len(corrects) == 0:
        return {
            "nb_instances": 0,
            "corrects": [],
            "accuracy": 0,
        }
    nb_inst = len(corrects)
    acc = sum(corrects) / nb_inst
    return {"nb_instances": nb_inst, "corrects": corrects, "accuracy": acc, "warnings": warnings}


def get_tick(is_correct, warning):
    if is_correct:
        tick = "✅"
    elif not is_correct and warning == 0:
        tick = "❌"
    elif warning >= 3:
        tick = "💀"
    elif warning >= 2:
        tick = "⚠️"
    else:
        # small warning
        tick = "❕"
    return tick


def get_problem_ticks(results, model, problem):
    stat = get_problem_stats(results, model, problem)
    ticks = ""
    for i, correct in enumerate(stat["corrects"]):
        ticks += get_tick(correct, stat["warnings"][i])
    return ticks


def get_model_stats(results, model):
    res = results[model]
    nb_problems = len(res)
    problem_stats = {problem: get_problem_stats(results, model, problem) for problem in res.keys()}
    stats = {"problem_stats": problem_stats.copy()}
    stats["nb_problems"] = len(res)
    if nb_problems == 0:
        stats["avg_accuracy"] = 0
    else:
        stats["avg_accuracy"] = sum([stat["accuracy"] for stat in problem_stats.values()]) / nb_problems
    return stats


def model_stats_to_html(stats):
    problem_stats_html = []
    problem_stats_html.append(
        A(
            "📋",
            href="javascript:void(0)",
            cls="sidebar-item copy-button",
            onclick="copyToClipboard()",
        )
    )
    for problem, stat in sorted(stats["problem_stats"].items(), key=lambda x: x[0]):
        problem_full_name = str(problem)
        if problem in sources:
            problem_full_name += f" ({sources[problem]})"
        p = f"{problem_full_name}:{' '*(30-len(str(problem_full_name)))}"
        p += f"{stat['accuracy']*100:6.2f}% "
        p += f"({stat['nb_instances']} instances: "
        for i, correct in enumerate(stat["corrects"]):
            p += get_tick(correct, stat["warnings"][i])
        p += ")"
        print(p)
        problem_stats_html.append(P(p, cls="problem-stats"))
    stats_html = [
        P(f"Avg Acc: {stats['avg_accuracy']*100:.2f}% ({stats['nb_problems']} problems)", cls="strong"),
        Div(*problem_stats_html),
    ]
    return stats_html


def parse_messages_response(response):
    # This is a list of messages
    response_str = response[0]["content"]
    for i in range(1, len(response)):
        if response[i]["role"] == "assistant":
            response_str += "\n\n" + 30 * "=" + "Assistant" + 30 * "=" + "\n\n" + response[i]["content"]
        else:
            response_str += "\n\n" + 30 * "=" + "User" + 30 * "=" + "\n\n" + response[i]["content"]
    return response_str


def sanitize_response(response):
    response = response.replace("\\( ", "$")
    response = response.replace(" \\)", "$")
    response = response.replace("\\(", "$")
    response = response.replace("\\)", "$")

    response = response.replace("\\[ ", "$$")
    response = response.replace(" \\]", "$$")
    response = response.replace("\\[", "$$")
    response = response.replace("\\]", "$$")
    return response


###### results


@rt("/expand/{url}/{expanded}")
def get(url: str, expanded: bool):
    global boxes_expanded
    boxes_expanded = expanded
    # This endpoint redirects and uses JavaScript to inject CSS that removes max-height
    # Since we can't persist state in a stateless redirect, we'll use a query parameter instead
    if url == "" or url is None:
        return Redirect(f"/")
    url = "/view/" + url.replace(">>>", "/")
    return Redirect(url)


@rt("/refresh/{comp}/{url}")
def get(comp: str, url: str):
    global current_comp
    current_comp = comp.replace("---", "/")
    global results
    results = analyze_run(current_comp, args.models)
    global sources
    sources = load_sources(current_comp)
    # results = analyze_run(run_dir, args.models, args.problems)[0]
    print("Refreshed!")
    if url == "" or url is None:
        return Redirect("/")
    url = "/view/" + url.replace(">>>", "/")
    return Redirect(url)


@rt("")
def index():
    # add dropdown with all comps
    dropdown = Select(
        *[Option(comp, value=comp, selected=(comp == current_comp)) for comp in all_comps],
        id="comp-dropdown",
        cls="sidebar-item",
        name="comp-dropdown",
    )

    # add button that calls /refresh
    sidebar_contents = [
        dropdown,
        A(
            "[Reload All Data]",
            href=f"/refresh/{current_comp.replace('/', '---')}",
            cls="sidebar-item reload-button strong",
        ),
        A("Home", href="/", cls="sidebar-item strong"),
    ]
    for model in results.keys():
        sidebar_contents.append(A(model, href=f"/view/{model}", cls="sidebar-item"))

    stats_html = []
    for model in results.keys():
        stats = get_model_stats(results, model)
        stats_html.append(H3(f"Model: {model}"))
        stats_html.append(Div(*model_stats_to_html(stats)))

    return Titled(
        title,
        Div(
            Div(*sidebar_contents, cls="sidebar"), Div(Div(*stats_html), cls="main"), style="display: flex; width: 100%"
        ),
    )


@rt("/view/{model}")
def get(model: str):
    # add dropdown with all comps
    dropdown = Select(
        *[Option(comp, value=comp, selected=(comp == current_comp)) for comp in all_comps],
        id="comp-dropdown",
        cls="sidebar-item",
        name="comp-dropdown",
    )
    print("model: ", model)
    links = [
        dropdown,
        A(
            "[Reload All Data]",
            href=f"/refresh/{current_comp.replace('/', '---')}/{model}",
            cls="sidebar-item reload-button strong",
        ),
        A("Home", href="/", cls="sidebar-item strong"),
    ]
    links.append(A(f"  {model}", href=f"/view/{model}", cls="sidebar-item strong current"))

    for problem in sorted(results[model].keys(), key=lambda x: int(x)):
        ticks = get_problem_ticks(results, model, problem)
        problem_full_name = str(problem)
        if problem in sources:
            problem_full_name += f" ({sources[problem]})"
        link_text = f"{problem_full_name} {ticks}"
        links.append(A(link_text, href=f"/view/{model}/{problem}", cls="sidebar-item"))

    stats = get_model_stats(results, model)
    stats_html = Div(*model_stats_to_html(stats))

    global boxes_expanded
    d = Div(*links[3:], cls=f"sidebar-list expanded-{boxes_expanded}")

    sidebar = Div(*links[:3], d, cls="sidebar")
    return Titled(
        title,
        Div(
            sidebar,
            Div(H3(f"Model: {model}", style="text-align: left;"), Div(stats_html), cls="main"),
            style="display: flex; width: 100%",
        ),
    )


def render_message(message):
    boxes = []
    # Prep tagline
    role = message["role"]
    if role == "developer":
        tagline = "System Prompt / Developer Message"
    elif role == "user":
        tagline = "User"
    elif role == "tool_response":
        tool_name = message["tool_name"]
        tool_call_id = message["tool_call_id"]
        tagline = f"Response from Tool {tool_name} (Tool Call ID: {tool_call_id})"
    elif role == "assistant":
        typ = message.get("type")
        if typ == "cot":
            tagline = "Assistant (Chain-of-Thought)"
        elif typ == "response":
            tagline = "Assistant"
        elif typ == "tool_call":
            tool_name = message["tool_name"]
            tool_call_id = message["tool_call_id"]
            tagline = f"Assistant (Tool Call to {tool_name}, Tool Call ID: {tool_call_id})"
        elif typ == "internal_tool_call":
            tool_name = message["tool_name"]
            assert tool_name == "code_interpreter"
            tagline = f"Assistant (Internal Tool Call to {tool_name})"
            # NOTE: only code_interpreter supported
        else:
            raise ValueError(f"Unknown assistant type: {typ}")
    else:
        raise ValueError(f"Unknown role: {role}")

    # Prep content
    if role == "assistant" and typ in ["tool_call", "internal_tool_call"]:
        if typ == "tool_call":
            arguments = message.get("arguments", {})
            for k, v in arguments.items():
                if k != "code":
                    tagline += f" ({k}: {v})"
            if "code" in arguments:
                code = arguments["code"]
            else:
                code = None
        else:
            code = message.get("code", None)

        if code is not None:
            code = code.replace("```python", "").replace("```", "").strip()
            code = Pre(Code(code), cls=f"language-python marked box response-box code")
            content = code
    else:
        content = message.get("content", "")
        if isinstance(content, str):
            content = Div(content, cls=f"marked box response-box {role}")
        else:
            # Fish for image and text
            assert isinstance(content, list)
            text, img = None, None
            for c in content:
                if c["type"] in ["text", "input_text"]:
                    text = c["text"]
                elif c["type"] == "input_image":
                    img = c["image_url"]
                elif c["type"] == "image_url":
                    img = c["image_url"]["url"]

            content = []
            if text is not None:
                content.append(text)
            if img is not None:
                content.append(Img(src=img, cls="problem-image"))
            content = Div(*content, cls=f"marked box response-box {role} conversation-content")

    # Return boxes
    boxes.append(Div(tagline, cls="strong", style="margin-top: 1rem; margin-bottom: 0.5rem;"))
    boxes.append(content)
    return boxes


@rt("/modelinteraction/{id}")
def get(id: str):  # model>>problemname>>id
    """
    See normalize_conversation in utils.py for the format
    """
    boxes = []
    tokens = id.split(">>")
    if len(tokens) == 4:
        model, problem_name, i, extra = tokens
        assert extra == "history"
        history = results[model][int(problem_name)]["history"][int(i)]

        # Create dropdown with all steps
        dropdown_options = []
        dropdown_options.append(Option("Select a step...", value=""))
        for step in history:
            stp = step["step"]
            timestep = step["timestep"]
            dropdown_options.append(
                Option(f"TIME={timestep} 🕐 {stp}", value=f"{model}>>{problem_name}>>{i}>>{timestep}>>{stp}")
            )

        dropdown = Select(*dropdown_options, id="history-step-selector", onchange="loadHistoryStep(this.value)")
        boxes.append(dropdown)

        # Container for dynamically loaded step content
        boxes.append(Div(id="history-step-content", style="margin-top: 1rem;"))

    else:
        model, problem_name, i = tokens
        conversation = results[model][int(problem_name)]["messages"][int(i)]
        for message in conversation:
            boxes.extend(render_message(message))
    return Div(*boxes)


@rt("/historystep/{id}")
def get(id: str):  # model>>problemname>>i>>timestep>>step
    """
    Returns the conversation for a specific history step
    """
    tokens = id.split(">>")
    model, problem_name, i, timestep_str, step_name = tokens
    timestep = int(timestep_str)

    history = results[model][int(problem_name)]["history"][int(i)]

    # Find the step with matching timestep and step name
    target_step = None
    for step in history:
        if step["timestep"] == timestep and step["step"] == step_name:
            target_step = step
            break

    if target_step is None:
        return Div("Step not found", cls="error")

    boxes = []

    conversation = target_step["messages"]
    for message in conversation:
        boxes.extend(render_message(message))

    return Div(*boxes)


@rt("/view/{model}/{problem_name}")
def get(model: str, problem_name: str):
    # add dropdown with all comps
    dropdown = Select(
        *[Option(comp, value=comp, selected=(comp == current_comp)) for comp in all_comps],
        id="comp-dropdown",
        cls="sidebar-item",
        name="comp-dropdown",
    )
    global boxes_expanded
    print("model: ", model, "problem_name: ", problem_name, "boxes_expanded: ", boxes_expanded)
    if boxes_expanded:
        toggle_text = "[Make Boxes Scrollable]"
    else:
        toggle_text = "[Make Boxes Very Tall]"
    links = [
        dropdown,
        A(
            toggle_text,
            href=f"/expand/{model}>>>{problem_name}/{not boxes_expanded}",
            cls="sidebar-item reload-button strong",
        ),
        A(
            "[Reload All Data]",
            href=f"/refresh/{current_comp.replace('/', '---')}/{model}>>>{problem_name}",
            cls="sidebar-item reload-button strong",
        ),
        A("Home", href="/", cls="sidebar-item strong"),
    ]
    links.append(A(f"  {model}", href=f"/view/{model}", cls="sidebar-item strong"))
    for problem in sorted(results[model].keys(), key=lambda x: int(x)):
        ticks = get_problem_ticks(results, model, problem)
        cls = "sidebar-item" if problem != problem_name else "sidebar-item current"
        problem_full_name = str(problem)
        if problem in sources:
            problem_full_name += f" ({sources[problem]})"
        link_text = f"{problem_full_name} {ticks}"
        links.append(A(link_text, href=f"/view/{model}/{int(problem)}", cls=cls))
    ticks = get_problem_ticks(results, model, problem_name)  # my ticks

    res = results[model][int(problem_name)]
    instances_html = []

    # # Read the problem description from data/{comp}.csv
    # with open(f"data/{args.comp}/problems.csv", "r") as f:
    #     reader = csv.reader(f)
    #     problems = [row for row in reader][1:]

    problem_statement = res["problem"]

    instances_html = []
    problem_idx = int(problem_name)
    img_path = f"/data/{current_comp}/problems/{problem_idx}.png"
    if os.path.exists(img_path[1:]):
        instances_html.append(
            Div(problem_statement, Img(src=img_path, cls="problem-image"), cls="marked box problem-box")
        )
    else:
        instances_html.append(Div(problem_statement, cls="marked box problem-box"))

    solution = res["gold_answer"]
    instances_html.append(Div(solution, cls="marked box solution-box"))

    for i, messages in enumerate(res["messages"]):
        curr_html = []
        # Lazy population
        # extras = {'id': f"{model}>>{problem_name}"}

        # curr_html.append(Details(Summary("Model Interaction:"), cls="response-box-details strong", **extras))

        # if not is_correct:
        #     curr_html.append(P(f"Parsecheck Details:", cls="strong"))
        #     curr_html.append(Div(parsecheck_details, cls=f"box details-box {correct_cls}"))

        answer, is_correct = res["answers"][i], res["correct"][i]
        warning = False
        if "warnings" in res:
            warning = res["warnings"][i]
        if answer is None:
            answer = "No answer found in \\boxed{}. Model was instructed to output answer in \\boxed{}."
        verdict = get_tick(is_correct, warning)
        print(verdict)
        correct_cls = "correct" if is_correct else "incorrect"

        curr_html.append(Div(cls=f"fake-hr"))

        history = results[model][int(problem_name)].get("history", None)
        if history is not None and len(history) > 0 and history[0] is not None:
            extras = {"id": f"{model}>>{problem_name}>>{i}>>history"}
            curr_html.append(Details(Summary(f"[Run {i}] History:"), cls="response-box-details strong", **extras))
        extras = {"id": f"{model}>>{problem_name}>>{i}"}
        curr_html.append(Details(Summary(f"[Run {i}] Conversation:"), cls="response-box-details strong", **extras))

        curr_html.append(P(f"[Run {i}] Parsed Answer ({verdict}, {warning} warnings):", cls="strong"))
        curr_html.append(Div(answer, cls=f"box answer-box {correct_cls}"))

        instances_html.append(Div(*curr_html))

    # for i, entry in enumerate(res):
    #     try:
    #         problem_class = problem_classes[problem_name]
    #         instance = problem_class.from_json(entry["problem"])
    #     except Exception as e:
    #         print(f"Error parsing an instance of the problem {problem_name}: {e}")
    #         continue
    #     answer, is_correct = entry["answer"], entry["is_correct"]
    #     verdict = "✅" if is_correct else "❌"
    #     correct_cls = "correct" if is_correct else "incorrect"
    #     parsecheck_details = entry["parsecheck_details"]

    #     curr_html = [Div(cls=f"fake-hr")]
    #     orig_suffix = f" (Original)" if instance.is_original() else ""
    #     curr_html.append(H3(f"Problem Instance #{i}{orig_suffix}:", cls="strong problem-instance-p"))
    #     curr_html.append(Div(str(instance), cls="marked box problem-box"))

    #     formatting_instructions = instance.get_formatting()
    #     curr_html.append(P(f"Formatting Instructions:", cls="strong"))
    #     curr_html.append(Div(formatting_instructions, cls="marked box problem-box"))

    #     solution = instance.get_solution()
    #     if solution is not None:
    #         curr_html.append(P(f"Our Solution:", cls="strong"))
    #         curr_html.append(Div(solution, cls="marked box solution-box"))

    #     # Lazy population
    #     extras = {'id': f"{model}>>{problem_name}>>{i}"}
    #     curr_html.append(Details(Summary("Model Interaction:"), cls="response-box-details strong", **extras))

    #     curr_html.append(P(f"Parsed Answer ({verdict}):", cls="strong"))
    #     curr_html.append(Div(answer, cls=f"box answer-box {correct_cls}"))
    #     if not is_correct:
    #         curr_html.append(P(f"Parsecheck Details:", cls="strong"))
    #         curr_html.append(Div(parsecheck_details, cls=f"box details-box {correct_cls}"))

    #     instances_html.append(Div(*curr_html))

    # Add script to scroll to current item
    mathjaxsetup = Script(
        """
        window.MathJax = {
        tex: {
            inlineMath: [['$', '$']]
        }
        };
    """
    )
    mathjax = Script(id="MathJax-script", src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js")
    scroll_and_lazyfetch_script = Script(
        """
        document.addEventListener('DOMContentLoaded', function() {
            const current = document.querySelector('.sidebar-item.current');
            if (current) {
                current.scrollIntoView({ behavior: 'auto', block: 'center' });
            }
        });

        document.addEventListener('DOMContentLoaded', function() {
            document.querySelectorAll('.response-box-details').forEach(function(element) {
                element.addEventListener('toggle', function(event) {
                    if (event.target.open) { // Check if the <details> is being opened
                        const idd = event.target.getAttribute('id');
                        if (!event.target.hasAttribute('data-loaded')) {
                            fetch(`/modelinteraction/${idd}`)  // Assuming you have a backend route to handle this
                                .then(response => response.text())
                                .then(data => {
                                    const wrapper = document.createElement('div');
                                    wrapper.className = 'conversation-content';
                                    wrapper.innerHTML = data;
                                    event.target.appendChild(wrapper);
                                    event.target.setAttribute('data-loaded', true); // Mark as loaded
                                    MathJax.typesetPromise();

                                    hljs.highlightAll();
                                })
                                .catch(error => console.error('Error fetching response details:', error));
                        }
                    }
                });
            });
        });

        function loadHistoryStep(stepId) {
            if (!stepId) {
                document.getElementById('history-step-content').innerHTML = '';
                return;
            }

            fetch(`/historystep/${stepId}`)
                .then(response => response.text())
                .then(data => {
                    document.getElementById('history-step-content').innerHTML = data;
                    MathJax.typesetPromise();
                    hljs.highlightAll();
                })
                .catch(error => {
                    console.error('Error fetching history step:', error);
                    document.getElementById('history-step-content').innerHTML = '<div class="error">Error loading step</div>';
                });
        }
    """
    )

    highlightjs_css = Link(
        rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/styles/default.min.css"
    )

    highlightjs = Script(
        id="HighlightJS", src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/highlight.min.js"
    )

    d = Div(*links[3:], cls=f"sidebar-list expanded-{int(boxes_expanded)}")
    sidebar = Div(*links[:3], d, cls="sidebar")
    problem_full_name = str(problem_name)
    if int(problem_name) in sources:
        problem_full_name += f" ({sources[int(problem_name)]})"

    # Add CSS to remove max-height if expanded parameter is True
    extra_elements = [mathjaxsetup, mathjax, highlightjs_css, highlightjs, sidebar, scroll_and_lazyfetch_script]
    if boxes_expanded:
        expand_style = Style(
            """
            .box {
                max-height: none !important;
            }
        """
        )
        extra_elements.insert(0, expand_style)

    return Titled(
        title,
        Div(
            *extra_elements,
            Div(
                H3(f"Model: {model}", style="text-align: left;"),
                H3(f"Problem: {problem_full_name} {ticks}", style="text-align: left;"),
                *instances_html,
                cls="main",
            ),
            style="display: flex; width: 100%",
        ),
    )


###
serve(reload=True, port=args.port)
