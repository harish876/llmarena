import argparse
import os
import json
import subprocess
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument("--comp", default="usamo/usamo_2024")
args = parser.parse_args()

# go over all folders in latex/{args.comp} recursively. Only look at ones that do not have subfolders anymore

def get_subfolders(path):
    all_folders = [f.path for f in os.scandir(path) if f.is_dir()]
    if len(all_folders) == 0:
        return [path]
    subfolders = []
    for folder in all_folders:
        subfolders += get_subfolders(folder)
    return subfolders

subfolders = get_subfolders(f"latex/{args.comp}")

for subfolder in subfolders:
    # get index of problem
    idx = int(subfolder.split("/")[-1])
    # find the pdf file in the subfolder
    file = [f for f in os.listdir(subfolder) if f.endswith(".pdf")]
    if len(file) == 0:
        logger.error(f"No pdf file found for problem {idx} in {subfolder}")
        continue

    file = file[0]
    # copy the pdf file to latex/{args.comp}/{idx}/pdf_name.pdf
    os.makedirs(f"latex/{args.comp}/{idx}", exist_ok=True)
    os.system(f"cp {subfolder}/{file} latex/{args.comp}/{idx}/{file}")
    logger.info(f"Copied {file} to latex/{args.comp}/{idx}/{file}")

