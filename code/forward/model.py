import time
import os
import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, AutoConfig

import warnings
import numpy as np



class Model:
    def __init__(self, model_name, device, num_gpus=1, run_svd=False, max_length=8000, half_precision=False):
        self.model_name = model_name
        self.device = device
        self.layer_index = None
        self.run_svd = run_svd
        self.half_precision = half_precision
        self.max_length = max_length
        self.num_gpus = num_gpus
        self.model, self.tokenizer = self.load_model(model_name)

    def load_model(self, model_name):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.bfloat16, "offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")

        print('Loading Model and Tokenizer from', model_name)
        access_token = 'TOKENHERE'
        config = AutoConfig.from_pretrained(model_name, token=access_token)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
        model = AutoModelForCausalLM.from_pretrained(model_name,
        low_cpu_mem_usage=True, config=config, use_auth_token=access_token, **kwargs)

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()
        
        return model, tokenizer

    def apply_svd(self, mat, k=1024):
        mat_mean = torch.mean(mat, dim=0)
        mat_centered = mat - mat_mean
        
        U, S, V = torch.svd(mat_centered, some=True)
        
        reduced_mat = torch.mm(mat_centered, V[:, :k])
        return reduced_mat


    def lm_layers(self, input_text, layers=None, layer_wise=False):
        with torch.no_grad():
            self.tokenizer.pad_token = self.tokenizer.eos_token

            DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

            messages = [
                {"role": "system", "content": "You are a precise information extractor. Directly give out the answer, make your answer as short as possible."},
                {"role": "user", "content": input_text},
            ]

            prompt = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True,tokenize=False
            )

            encoding = self.tokenizer(
                prompt, return_tensors="pt").to(self.device)
            outputs = self.model(**encoding, output_hidden_states=True)

            hidden_layers = outputs.hidden_states
            

            if not layer_wise:
                layers_mean = []
                layers_last = []
                for layer_index in layers:
                    layer = hidden_layers[layer_index].squeeze(0)
                    layer_last = layer[-1, :]


                    layer_mean = layer.mean(dim=0)
                    if self.half_precision:
                        layer_last = layer_last.reshape(-1).to(torch.bfloat16).cpu()
                        layer_mean = layer_mean.reshape(-1).to(torch.bfloat16).cpu()
                    else:
                        layer_last = layer_last.reshape(-1).to(torch.float32).cpu()
                        layer_mean = layer_mean.reshape(-1).to(torch.float32).cpu()
                    layers_mean.append(layer_mean)
                    layers_last.append(layer_last)
                layers_mean_mat = torch.stack(layers_mean, dim=0)
                layers_last_mat = torch.stack(layers_last, dim=0)
                return layers_mean_mat, layers_last_mat
            else:
                layers_out = {}
                layers_last = {}
                for layer_index in layers:
                    layer = hidden_layers[layer_index].squeeze(0)
                    layer_last = layer[-1,:]
                    layer = layer.mean(dim=0)
                    if self.half_precision:
                        layer = layer.reshape(-1).to(torch.bfloat16).cpu()
                        layer_last = layer_last.reshape(-1).to(torch.bfloat16).cpu()
                    else:
                        layer = layer.reshape(-1).to(torch.float32).cpu()
                        layer_last = layer_last.reshape(-1).to(torch.float32).cpu()
                    layers_out[layer_index] = layer

                    layers_last[layer_index] = layer_last

                return layers_out, layers_last

    def out_attention(self, input_text):
        with torch.no_grad():
            self.tokenizer.pad_token = self.tokenizer.eos_token

            DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

            messages = [
                {"role": "system", "content": "You are a precise information extractor. Directly give out the answer, make your answer as short as possible."},
                {"role": "user", "content": input_text},
            ]

            prompt = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True,tokenize=False
            )

            encoding = self.tokenizer(
                prompt, return_tensors="pt").to(self.device)
            
            outputs = self.model(**encoding, output_attentions=True)


            attentions = outputs.attentions

            return attentions

