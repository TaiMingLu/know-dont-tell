import argparse
import json
from tqdm import tqdm
import torch
import transformers
import torch
import string
from typing import List
from multi_doc import *
import regex

# Setup argparse to handle command-line arguments
parser = argparse.ArgumentParser(description="Process and generate answers using a text-generation model.")
parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model identifier for the transformer model.")
parser.add_argument('--data_path', type=str, default='/path/to/your/data.jsonl.gz', help="Path to the data file.")
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


index_map = {
    10: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    20: [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    30: [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29],
    40: [0, 3, 7, 11, 15, 18, 22, 27, 31, 35, 39],
    35: [0, 4, 7, 11, 14, 18, 21, 25, 28, 32, 34],
    45: [0, 5, 9, 13, 17, 21, 25, 29, 33, 37, 44],
    55: [0, 5, 11, 16, 22, 27, 32, 38, 43, 49, 54],
}

results = {}

for n in [10, 20, 30, 40, 50]:
    results[n] = {}
    for index in index_map[n]:
        results[n][index] = {'correct': 0, 'total': 0}
    num_data = 100
    docs = get_multi_doc_data(args.data_path, num_data, n, 1)
    for i in tqdm(range(num_data)):
        doc = docs[i]
        for index in index_map[n]:
            prompt = get_prompt(doc, index)

            messages = [
                {"role": "system", "content": "You are a precise information extractor. Directly give out the answer, make your answer as short as possible."},
                {"role": "user", "content": prompt},
            ]

            terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = pipeline(
                messages,
                max_new_tokens=50,
                eos_token_id=terminators,
                do_sample=False,
            )
            
            model_answer = outputs[0]["generated_text"][-1]['content']
            print(model_answer)

            results[n][index]['total'] += 1
            for answer in doc['answers']:
                print(answer)
                if normalize_answer(answer) in normalize_answer(model_answer):
                    results[n][index]['correct'] += 1
                    break
             
        
        with open('multi-docs.json', 'w') as file:
            json.dump(results, file, indent=4)