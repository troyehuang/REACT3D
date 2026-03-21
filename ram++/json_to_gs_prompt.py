import json
import argparse

parser = argparse.ArgumentParser(
    description="read json and generate grounded sam text prompt"
)
parser.add_argument(
    "--json_path",
    help="json file path containing tags",
    type=str,
    default="/home/troye/ssd/Zhao_SP/data/40aec5fffa/openable_objects.json",
)
args = parser.parse_args()

with open(args.json_path, "r", encoding="utf-8") as f:
    tags = json.load(f)

# 生成并打印提示
prompt = ". ".join(tags) + "."
print(prompt)