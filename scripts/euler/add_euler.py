import argparse
from datetime import datetime
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--problem_id", type=int, required=True)
args = parser.parse_args()

import glob
import os

problems_dir = "data/euler/euler/problems"
tex_files = glob.glob(os.path.join(problems_dir, "*.tex"))

next_idx = len(tex_files) + 1
new_path = f"data/euler/euler/problems/{next_idx}.tex"
print(f"Found {len(tex_files)} problem files, next problem id: {next_idx}")
assert not os.path.exists(new_path)

# write the content
url = f"https://projecteuler.net/minimal={args.problem_id}"
problem_content = requests.get(url).text
with open(new_path, "w") as f:
    f.write(problem_content)

# add the id to the list of ids
import csv

source_path = "data/euler/euler/source.csv"
with open(source_path, "a", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'source'])
    writer.writerow({'id': next_idx, 'source': f"euler{args.problem_id}"})

# add dummy answer to the list of answers
euler_answers_path = "data/euler/euler/answers.csv"
with open(euler_answers_path, "a", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'answer'])
    writer.writerow({'id': next_idx, 'answer': "none"})

# increase number of problems in the config
import yaml

config_path = "configs/competitions/euler/euler.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
config['n_problems'] = next_idx
# today's date formatted as YYYY-MM-DD
config['date'][str(args.problem_id)] = f"{datetime.now().strftime("%Y-%m-%d")}?" # check!

with open(config_path, "w") as f:
    yaml.safe_dump(config, f)

# increase number of problems in the website config
website_config_path = "website/flaskr/static/data/competitions.json"
import json

with open(website_config_path, "r") as f:
    website_config = json.load(f)
website_config['competition_info']['euler--euler']['num_problems'] = next_idx
website_config['competition_info']['euler--euler']['problem_names'].append(str(args.problem_id))
website_config['competition_info']['euler--euler']['problem_difficulty'][str(args.problem_id)] = "todo"
with open(website_config_path, "w") as f:
    json.dump(website_config, f, indent=4)

print("!!! Check the problem date in yaml (added as today)")
print("!!! Problem difficulty needs to be set manually (added as 'todo')")