from model import Model
import argparse
import json
from tqdm import tqdm
import os
import torch
import uuid
import copy

# Create the parser
parser = argparse.ArgumentParser(description="Process the parameters for saving each layer's last token embedding.")

parser.add_argument('--model_name', type=str, default='lmsys/longchat-7b-16k', 
                    help='Name of the model to be loaded (default: lmsys/longchat-7b-16k)')
parser.add_argument('--indices', type=int, nargs='+', default=[1, 25, 50, 75], 
                    help='List of indices to process (default: [1, 25, 50, 75])')
parser.add_argument('--num_kv', type=int, default=100, 
                    help='Number of kv pair in the a single prompt (default: 75)')
parser.add_argument('--layers', type=int, nargs='+', default=[1], 
                    help='List of indices to process (default: [1], for our model, from 1(first hidden layer) to 32(output layer, total 33 layers))')
parser.add_argument('--save_path', type=str, required=True,
                    help='Path where the outputs will be saved (no default, required to specify)')
parser.add_argument('--model_path', type=str, default='.cache/huggingface/hub', 
                    help='Name of the model to be cached (default: .cache/huggingface/hub)')
parser.add_argument('--num_data', type=int, default=2000,
                    help='Optional: Number of total data before index permutation (default: 2000)')
parser.add_argument('--all_layers', action='store_true',
                    help='Include all layers in the processing.')
parser.add_argument('--all_indices', action='store_true',
                    help='Include all indices in the prompt.')
args = parser.parse_args()

if args.all_layers:
    args.layers = [i for i in range(1, 33)]

if args.all_indices:
    args.indices = [i for i in range(args.num_kv)]

# Example usage of the parsed arguments
print(f"Model Name: {args.model_name}")
print(f"Indices: {args.indices}")
print(f"Layers: {args.layers}")
print(f"Data Size: {args.num_data}")
print(f"Save Path: {args.save_path}")

for layer in args.layers:
    assert layer >= 1 and layer <= 32, 'We only want layer index between 1 and 33'
for index in args.indices:
    assert index >= 0 and index <= args.num_kv, 'Index should in valid values'

def get_kv_paris(n):
    return [[str(uuid.uuid4()) for _ in range(n)], [str(uuid.uuid4()) for _ in range(n)]]

def get_prompt(kv_paris, index):
    prompt = f'Extract the value corresponding to the specified key in the JSON object below.\n\nKey: "{kv_paris[0][index-1]}"\n\nJSON data:\n{{'
    for i in range(len(kv_paris[0]) - 1):
        prompt += f'"{kv_paris[0][i]}": "{kv_paris[1][i]}"\n'
    prompt += f'"{kv_paris[0][-1]}": "{kv_paris[1][-1]}"'
    prompt += '}\n\n'
    prompt += f'Key: "{kv_paris[0][index-1]}"\nCorresponding value:'

    return {'prompt': prompt, 'index': index, 'value': kv_paris[1][index-1]}

model = Model(args.model_name, 'cuda')

model_name = args.model_name.split('/')[-1]
os.makedirs(os.path.join(args.save_path, model_name), exist_ok=True)
os.makedirs(os.path.join(args.save_path, model_name, 'data'), exist_ok=True)
os.makedirs(os.path.join(args.save_path, model_name, str(args.num_kv)), exist_ok=True)

data = []
save_index = 1
for i in tqdm(range(args.num_data), desc="Prompts", position=0):
    kv_paris = get_kv_paris(args.num_kv)
    for index in args.indices:
        prompt = get_prompt(kv_paris, index)
        _, layers_mat = model.lm_layers(prompt['prompt'], args.layers)
        data.append((copy.deepcopy(layers_mat), prompt['value'], index,kv_paris))

        del layers_mat
        torch.cuda.empty_cache()
    
        if len(data) >= (args.num_data * len(args.indices) / 10):
            torch.save(data, os.path.join(args.save_path, model_name,str(args.num_kv), f'subset_{args.num_data}_{save_index}.pth'))
            save_index += 1
            data = []

print('Done Probing')












