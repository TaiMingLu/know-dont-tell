import argparse
import json
from tqdm import tqdm
import torch
import uuid
import transformers
import torch
import string
from typing import List
import regex

parser = argparse.ArgumentParser(description="Process and generate answers using a text-generation model.")
parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model identifier for the transformer model.")
parser.add_argument('--num_data', type=int, default=1000, help="Number of data to test.")
args = parser.parse_args()

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


model_id = args.model

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",

)


def get_kv_paris(n):
    return [[str(uuid.uuid4()) for _ in range(n)], [str(uuid.uuid4()) for _ in range(n)]]

def get_prompt(kv_pairs, index):
    prompt = "Extract the value corresponding to the specified key in the JSON object below.\nJSON data:\n{"
    for i in range(len(kv_pairs[0]) - 1):
        prompt += f'"kvpair[{i}]{kv_pairs[0][i]}": "{kv_pairs[1][i]}",\n'
    prompt += f'"kvpair[{len(kv_pairs[0]) - 1}]{kv_pairs[0][-1]}": "{kv_pairs[1][-1]}"'
    prompt += '}\n'
    prompt += f'Key: "{kv_pairs[0][index]}"\nCorresponding value:'

    return {'prompt': prompt, 'index': index, 'value': kv_paris[1][index]}

index_map = {
    20: [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    40: [0, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39],
    60: [0, 5, 11, 17, 23, 29, 35, 41, 47, 53, 59],
    80: [0, 7, 15, 23, 31, 39, 47, 55, 63, 71, 79],
    100: [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99],
}

results = {}

for n in [20, 40, 60, 80, 100]:
    results[n] = {}
    for index in index_map[n]:
        results[n][index] = {'correct': 0, 'total': 0}
    for _ in range(args.num_data):
        kv_paris = get_kv_paris(n)
        for index in index_map[n]:
            prompt = get_prompt(kv_paris, index)
            print(prompt, flush=True)

            messages = [
                {"role": "system", "content": "You are a very precise long information retrival machine. Directly give out the answer from the given data, make your answer as short as possible."},
                {"role": "user", "content": prompt['prompt']},
            ]

            terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = pipeline(
                messages,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=False,
            )
            
            answer = outputs[0]["generated_text"][-1]['content']
            print(answer, flush=True)

            if normalize_answer(prompt['value']) in normalize_answer(answer):
                print(normalize_answer(prompt['value']), flush=True)
                results[n][index]['correct'] += 1
            results[n][index]['total'] += 1
        
        with open('kv_paris.json', 'w') as file:
            json.dump(results, file, indent=4)