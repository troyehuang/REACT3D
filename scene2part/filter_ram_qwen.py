import argparse
import os
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INSTRUCTION = """
```
You are an indoor scene openable object label filter.
I will give you many object labels. For the labels I give to you, you need to filter them and only give me the labels from the input that are openable objects.

Openable object must satisfy ALL conditions:
1) It has a **DOOR** component that humans can open in daily life.
2) It has either a **REVOLUTE** joint or a **PRISMATIC** joint.

Examples of openable objects: cabinet, drawer, microwave.

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

    # cleaned, seen = [], set()
    # for it in raw_items:
    #     it_norm = normalize(it)
    #     if not it_norm.endswith(" door"):
    #         continue
    #     base = re.sub(r"\s+door$", "", it_norm).strip()
    #     if base in input_norm and it_norm not in seen:
    #         cleaned.append(it_norm)   # preserve the normalized version
    #         seen.add(it_norm)

    cleaned, seen = [], set()
    for it in raw_items:
        it_norm = normalize(it)
        
        if it_norm.endswith(" door"):
            base = re.sub(r"\s+door$", "", it_norm).strip()
        else:
            base = it_norm
            
        if base in input_norm:
            final_item = f"{base} door"
            if final_item not in seen:
                cleaned.append(final_item)
                seen.add(final_item)

    # If the bare "door" item is not present, supplement it
    if "door" not in seen:
        cleaned.insert(0, "door")
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    
    # ---------------------------------------------------------
    # Model Loading
    # ---------------------------------------------------------
    model_name = "Qwen/Qwen3.5-4B"
    
    print(f"Loading tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # ---------------------------------------------------------
    # Data Loading
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Qwen3 Inference
    # ---------------------------------------------------------
    messages = [
        {"role": "user", "content": prompt_text}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        temperature=0.0
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("Generating response...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=8192,
        do_sample=False
    )
    
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    print("Response:", response)
    # ---------------------------------------------------------
    # Post-Processing
    # ---------------------------------------------------------
    cleaned_items, cleaned_dot = parse_and_clean_response(response, class_names)

    print("\nCleaned dotline:", cleaned_dot)

    # Save all responses and results
    with open(save_txt, "w", encoding="utf-8") as f:
        f.write(cleaned_dot)
        
    print(f"\nSaved successfully to {save_txt}")

if __name__ == "__main__":
    main()
