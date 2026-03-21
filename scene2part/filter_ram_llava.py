import argparse
import torch
import os
import cv2
import numpy as np
import subprocess
import time
import json

from llava.eval.run_llava import eval_model_no_load, eval_model_text_only
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re

INSTRUCTION = """
```
You are an indoor scene openable object class detector.
I will give you many object class names. For the class names I give to you, you need to filter them and only give me the class names from the input that are openable objects.

Openable object must satisfy ALL conditions:
1) It has a door-like component that humans can open in daily life.
2) It has either a revolute joint or a prismatic joint.

Examples of openable objects: cabinet, drawer, microwave.

For each openable object you filter, append the word "door" AFTER the original class name. Examples:
- input "microwave" -> output "microwave door"
- input "cabinet"   -> output "cabinet door"
- input "drawer"   -> output "drawer door"
- input "fridge"   -> output "fridge door"

CRITICAL CONSTRAINTS:
- Only use class names that appear in the input list. Do NOT invent new names.
- Output format: return ONLY a single line string in this exact format:
  item1. item2. item3.
That is: each item ends with a period and a single space; the last item also ends with a period. No extra text, no code fences, no JSON, no bullets, no explanations.
```
"""
def ensure_dotline_format(text: str) -> bool:
    """
    must match format like: 'item1. item2. item3.'
    """
    line = text.strip()
    if "\n" in line:
        return False
    
    return re.fullmatch(r'[^.]+\.(?: [^.]+\.)*', line) is not None

def parse_and_clean_response(resp_text: str, input_classes: list):
    """
    Rules:
    1) Regular items must end with ' door', and the base without ' door' must be in input_classes
    2) Deduplicate and maintain order
    3) After cleaning, if there is no exact 'door' item, append 'door' to the end of the list
    4) Finally, check input_classes, supplement normalized classes ending with ' door' to cleaned (if not already there)
    Returns: cleaned_items(list[str]), cleaned_dotline(str)
    """
    raw_items = extract_items_from_dotstring(resp_text)

    input_norm = {normalize(x) for x in input_classes}

    cleaned, seen = [], set()
    for it in raw_items:
        it_norm = normalize(it)
        if not it_norm.endswith(" door"):
            continue
        base = re.sub(r"\s+door$", "", it_norm).strip()
        if base in input_norm and it_norm not in seen:
            cleaned.append(it_norm)   # preserve the normalized version
            seen.add(it_norm)

    # If the bare "door" item is not present, supplement it
    if "door" not in seen:
        cleaned.append("door")
        seen.add("door")

    # Finally check input_classes, supplement normalized classes ending with ' door' to cleaned
    for cls in input_classes:
        cls_norm = normalize(cls)
        if cls_norm.endswith(" door") and cls_norm not in cleaned:
            cleaned.append(cls_norm)

    dotline = list_to_dotstring(cleaned)

    if not ensure_dotline_format(dotline):
        raise ValueError("Sanity check failed: cleaned dotline is not in 'item. item. ...' format.")

    return cleaned, dotline

def list_to_dotstring(items):
    items = [s.strip() for s in items if str(s).strip()]
    if not items:
        return ""
    return ". ".join(items) + "."

def extract_items_from_dotstring(text):
    """
    Extract tokens like 'xxx.' from free text, ignoring any surrounding chatter.
    Returns a list of clean strings without the trailing dot.
    """
    # Grab everything that looks like "...something..." followed by a dot
    # (avoid newlines in tokens; trim whitespace).
    tokens = [t.strip() for t in re.findall(r'([^.\n]+)\.', text)]
    # Dedup while preserving order
    seen, out = set(), []
    for t in tokens:
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out

def normalize(s):
    return re.sub(r"\s+", " ", s.strip().lower())


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)

#model_path = "liuhaotian/llava-v1.5-7b"
#model_path = "liuhaotian/llava-v1.6-vicuna-7b"
model_path = "liuhaotian/llava-v1.6-mistral-7b"

disable_torch_init()

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name
)

args = parser.parse_args()
    
y_list = []
n_list = []
f_list = []
response_dict = {}

scene_tags_path = os.path.join(args.data_dir, "scene_tags.json")
save_txt = os.path.join(args.data_dir, "sam_prompt.txt")

if not os.path.isfile(scene_tags_path):
    raise FileNotFoundError(f"Cannot find {scene_tags_path}")
with open(scene_tags_path, "r", encoding="utf-8") as f:
    class_names = json.load(f)
if not isinstance(class_names, list):
    raise ValueError("scene_tags.json must be a JSON array of strings.")
input_dot = list_to_dotstring(class_names)

prompt_text = (
    INSTRUCTION
    + "\n\nHere are the input class names (dot-separated):\n"
    + input_dot
    + "\n"
)

model_args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt_text,
    "conv_mode": None,
    "image_file": None,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 8192,
})()

response = eval_model_text_only(model_args, tokenizer, model)
response = str(response)

#print("Response:", response)

cleaned_items, cleaned_dot = parse_and_clean_response(response, class_names)

print("Cleaned dotline:", cleaned_dot)

# Save all responses and results
with open(save_txt, "w", encoding="utf-8") as f:
    f.write(cleaned_dot)





