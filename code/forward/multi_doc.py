import argparse
import json
import logging
import sys
from copy import deepcopy
import random

from tqdm import tqdm
from xopen import xopen


def get_multi_doc_data(input_path, num_data, num_total_documents, variation):
    if num_total_documents < 2:
        raise ValueError(f"`num_total_documents` must be at least 2, got {num_total_documents}")

    # Validate that we have at least num_total_documents for every example
    count = 0
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            # print(line, flush=True)
            qa_retrieval_result = json.loads(line)
            example_num_documents = len([doc for doc in qa_retrieval_result["ctxs"] if doc["hasanswer"] is False])
            if num_total_documents > example_num_documents:
                raise ValueError(
                    f"Requested `num_total_documents` {num_total_documents}, but found an input"
                    f"example with only {example_num_documents} documents that don't contain the answer."
                )
            count += 1
            if count >= num_data:
                break


    data = []

    num_output_examples = 0
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            for _ in range(variation):
                qa_retrieval_result = json.loads(line)
                # Get documents that don't contain the answer
                valid_distractors_with_retrieval_indices = [
                    (idx, doc) for idx, doc in enumerate(qa_retrieval_result["ctxs"]) if doc["hasanswer"] is False
                ]
                
                # Take the top `num_total_documents - 1` distractors
                distractor_docs_with_retrieval_indices = deepcopy(
                    # valid_distractors_with_retrieval_indices[: num_total_documents - 1]
                    random.sample(valid_distractors_with_retrieval_indices, num_total_documents - 1)
                )
                for original_retrieval_index, distractor_doc in distractor_docs_with_retrieval_indices:
                    distractor_doc["original_retrieval_index"] = original_retrieval_index
                    distractor_doc["isgold"] = False
                distractor_docs = [x[1] for x in distractor_docs_with_retrieval_indices]

                content_selection_example = deepcopy(qa_retrieval_result)
                gold_chunk = {
                    "title": qa_retrieval_result["nq_annotated_gold"]["title"],
                    "text": qa_retrieval_result["nq_annotated_gold"]["chunked_long_answer"],
                    "hasanswer": True,
                    "isgold": True,
                }
                ctxs = distractor_docs
                
                content_selection_example["gold"] = gold_chunk

                content_selection_example["ctxs"] = ctxs
                
                data.append(content_selection_example)
            num_output_examples += 1

            if num_output_examples >= num_data:
                break

    print(f'Number of data Loaded: {len(data)}')
    len_dist = len(data[0]['ctxs'])
    print(f'Number of distractors per data: {len_dist}')
    print('___________________')
    # print(data[1])
    
    return data



def get_prompt(content, index):
    ctxs = deepcopy(content['ctxs'])
    gold = deepcopy(content['gold'])
    question = deepcopy(content['question'])
    ctxs.insert(index, gold)
    prompt = 'Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).\n\n'
    for i in range(len(ctxs)):
        doc = ctxs[i]
        title = doc['title']
        text = doc['text']
        prompt += f'Document [{i}] (Title: {title}) {text}\n'
    prompt += f'\nQuestion: {question}\nAnswer: '

    return prompt

