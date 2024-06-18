from model import Model
import argparse
import json
from tqdm import tqdm
import os
import torch
import uuid
import copy
from multi_doc import *

# Create the parser
parser = argparse.ArgumentParser(description="Process the parameters for saving each layer's last token embedding.")

# Add arguments to the parser
parser.add_argument('--model_name', type=str, default='lmsys/longchat-7b-16k', 
                    help='Name of the model to be loaded (default: lmsys/longchat-7b-16k)')
parser.add_argument('--data_path', type=str, help='path to documents data')

parser.add_argument('--indices', type=int, nargs='+', default=[1, 25, 50, 75], 
                    help='List of indices to process (default: [1, 25, 50, 75])')
parser.add_argument('--layers', type=int, nargs='+', default=[1], 
                    help='List of indices to process (default: [1], for our model, from 1(first hidden layer) to 32(output layer, total 33 layers))')
parser.add_argument('--all_layers', action='store_true',
                    help='Include all layers in the processing.')
parser.add_argument('--all_indices', action='store_true',
                    help='Include all indices in the prompt.')

parser.add_argument('--save_path', type=str, required=True,
                    help='Path where the outputs will be saved (no default, required to specify)')
parser.add_argument('--save_name', type=str, default=None,
                    help='Index to save the code.')

parser.add_argument('--num_data', type=int, default=2000,
                    help='Optional: Number of total data before index permutation (default: 2000)')
parser.add_argument('--num_variation', type=int, default=5,
                    help='Optional: Number of variation on data (default: 5)')
parser.add_argument('--data_length', type=int, default=50, 
                    help='Number of total documents in one single data (default: 50)')


args = parser.parse_args()

if args.all_layers:
    args.layers = [i for i in range(1, 33)]

if args.all_indices:
    args.indices = [i for i in range(args.data_length)]
else:
    args.indices = [i-1 for i in args.indices]


# Example usage of the parsed arguments
print(f"Model Name: {args.model_name}")
print(f"Indices: {args.indices}")
print(f"Layers: {args.layers}")
print(f"Data Size: {args.num_data}")
print(f"Num Documentation per data: {args.data_length}")
print(f"Save Path: {args.save_path}")


for layer in args.layers:
    assert layer >= 1 and layer <= 32, 'We only want layer index between 1 and 33'
for index in args.indices:
    assert index >= 0 and index < args.data_length, 'Index should in valid values'

data = get_multi_doc_data(args.data_path, args.num_data, args.data_length, args.num_variation) 


model = Model(args.model_name, 'cuda')

model_name = args.model_name.split('/')[-1]
os.makedirs(os.path.join(args.save_path, model_name), exist_ok=True)
os.makedirs(os.path.join(args.save_path, model_name, str(args.data_length)), exist_ok=True)

results = []
save_index = 1
for i in tqdm(range(len(data)), desc="Documents", position=0):
    documents = data[i]
    for index in args.indices:
        prompt = get_prompt(documents, index)

        _, layers_last_mat = model.lm_layers(prompt, args.layers)

        results.append((copy.deepcopy(layers_last_mat), i, index, documents['question'], documents['answers'], ))

        del layers_last_mat
        torch.cuda.empty_cache()

        if len(results) >= (args.num_data * args.num_variation * len(args.indices) / 10):
            torch.save(results, os.path.join(args.save_path, model_name,str(args.data_length), f'subset_{args.num_data * args.num_variation}_{save_index}.pth'))
            save_index += 1
            results = []


print('Done Probing')









