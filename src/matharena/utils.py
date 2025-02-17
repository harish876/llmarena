import re
from sympy.parsing.latex import parse_latex
import sympy
import sys

def latex2sympy_fixed(latex: str):
    # if _integer is present, replace it with _{integer} for any integer
    latex = re.sub(r"_([0-9]+)", r"_{\1}", latex)
    latex_parsed = parse_latex(latex)
    # replace constants like pi and e with their numerical value
    known_constants = {'pi': sympy.pi, 'e': sympy.E}

    # Replace any symbol in expr that is in our known_constants dictionary.
    expr = latex_parsed.xreplace({s: known_constants[s.name]
                        for s in latex_parsed.free_symbols
                        if s.name in known_constants})
    return expr
  
def get_latex_array(mat: list[list[str]]) -> str:
    n_cols = len(mat[0])
    cs = "{" + "c" * n_cols + "}"
    return "\\begin{array}" + cs + "\n" + "\n".join([" & ".join(row) + " \\\\" for row in mat]) + "\n\\end{array}"

def convert_to_int(obj):
    if isinstance(obj, str):
        return int(obj)
    if isinstance(obj, list):
        return [convert_to_int(item) for item in obj]
    raise ValueError(f"Cannot convert {type(obj)} to int")

def get_depth(solution):
    if isinstance(solution, (list, tuple)):
        depths = [
            get_depth(sol) for sol in solution
        ]
        if any([d is None for d in depths]):
            return None
        if len(set(depths)) > 1:
            return None
        return depths[0] + 1
    return 0