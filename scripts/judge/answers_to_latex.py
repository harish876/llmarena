import argparse
import os
import json
import subprocess
from loguru import logger
import re

parser = argparse.ArgumentParser()
parser.add_argument("--comp", default="usamo/usamo_2024")
args = parser.parse_args()

def get_subfolders(path):
    all_folders = [f.path for f in os.scandir(path) if f.is_dir()]
    if len(all_folders) == 0:
        return [path]
    subfolders = []
    for folder in all_folders:
        subfolders += get_subfolders(folder)
    return subfolders

subfolders = get_subfolders(f"outputs/{args.comp}")

for data_dir in subfolders:
    files = os.listdir(data_dir)

    data_dir_no_outputs = data_dir.replace("outputs", "latex")

    for file in files:
        answers_problem = json.load(open(f"{data_dir}/{file}"))
        anon_id = answers_problem["anonymous_id"]
        idx = answers_problem["idx"] + 1

        os.makedirs(f"{data_dir_no_outputs}/{idx}", exist_ok=True)

        # copy latex/main.tex to latex/{args.comp}/{base_config}/{idx}/{anon_id}.tex
        with open(f"latex/main.tex", "r") as f:
            content = f.read()
        with open(f"{data_dir_no_outputs}/{idx}/{anon_id}.tex", "w") as f:
            f.write(content)

        content = content[:content.rfind("\\begin{document}")]
        content += f"\\begin{{document}}\n\\input{{answers.tex}}\n\\end{{document}}"

        with open(f"{data_dir_no_outputs}/{idx}/{anon_id}.tex", "w") as f:
            f.write(content)

        # content for answers.tex
        content = "\\section{Problem}\n" + answers_problem["problem"] + "\n\n"

        for i, messages in enumerate(answers_problem["messages"]):
            content += f"\\subsection{{Answer Attempt {i + 1}}}\n"
            message = messages[-1]["content"]
            # ensure that every newline is a paragraph break
            message = re.sub(r'\r?\n-', '\n\n-', message)
            if "</think>" in message:
                message = message[message.index("</think>"):].replace("</think>", "")
            content += message + "\n\n"
        
        with open(f"{data_dir_no_outputs}/{idx}/answers.tex", "w") as f:
            f.write(content)
        
        # compile with pdflatex, check if it goes wrong
        output = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                f"{anon_id}.tex"
            ],
            cwd=f"{data_dir_no_outputs}/{idx}",  # set working directory here
            capture_output=True
        )
        stdout = str(output.stdout)
        if f"Output written on {anon_id}.pdf" not in stdout:
            output_std = output.stdout
            logger.error(f"Error compiling {data_dir_no_outputs} for problem {idx}")
        else:
            logger.info(f"Compiled {data_dir_no_outputs} for problem {idx}")
