import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--comp", default="aime/aime_2025_I")
args = parser.parse_args()

data_dir = f"data/{args.comp}/problems"

comp_no_space = args.comp.replace("/", "_")

os.makedirs("logs/latex", exist_ok=True)
with open(f"logs/latex/{comp_no_space}_merged.tex", "w") as f:
    file_list = list(os.listdir(data_dir))
    try:
        file_list = sorted(file_list, key=lambda x: int(x.split(".")[0]))
    except:
        pass
    for file in file_list:
        if not file.endswith(".tex"):
            continue
        f.write(f"\\subsection{{{file[:-4]}}}\n")
        with open(f"{data_dir}/{file}", "r") as g:
            f.write(g.read()) 
        f.write("\n\n")

# rewrite the main file
file_name = f"logs/latex/main.tex"

with open(file_name, "r") as f:
    content = f.read()

content = content[: content.rfind("\\begin{document}")]

content += f"\\begin{{document}}\n\\input{{{comp_no_space}_merged.tex}}\n\\end{{document}}"

with open(file_name, "w") as f:
    f.write(content)
