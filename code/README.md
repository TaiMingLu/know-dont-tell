### Environment Setup
```
conda create --n know_dont_tell python=3.9
cd code
conda activate know_dont_tell
pip install -r requirements.txt
```

### For Multi-Documents Question Answering
Downloaded the data from [Liu et al.](https://arxiv.org/abs/2307.03172)

```
wget https://nlp.stanford.edu/data/nfliu/lost-in-the-middle/nq-open-contriever-msmarco-retrieved-documents.jsonl.gz
```

## Probing Accuracy
### Multi-Documents Question Answering
```
cd forward
python inference_and_save_doc.py --model_name <MODEL_NAME> --data_path <JSON_DATA_PATH> --save_path <DATA_SAVE_PATH> --indices 1 3 6 9 12 15 18 21 24 27 30 --all_layers --num_variation 50 --num_doc 30 --num_data 2000
```
```
cd probe
python probe.py --data_path <DATA_PATH> --output_folder <PROBING_RESULTS_FOLDER>
```

### KV-Paris Retrieval
```
cd forward
python inference_and_save_kv.py --model_name <MODEL_NAME> --save_path <DATA_SAVE_PATH> --indices 1 10 20 30 40 50 60 70 80 90 100 --all_layers --num_kv 100 --num_data 20000
```
```
cd probe
python probe.py --data_path <DATA_PATH> --output_folder <PROBING_RESULTS_FOLDER>
```

## Generation Accuracy
### Multi-Documents Question Answering
```
cd generation
python doc.py --model <MODEL_NAME> --data_path <JSON_DATA_PATH> --num_data 1000
```

### KV-Paris Retrieval
```
python doc.py --model <MODEL_NAME> --num_data 10000
```


