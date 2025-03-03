import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, Dataset, ClassLabel, DatasetDict
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F 
import math
from collections import defaultdict
import tensorly as tl
from functools import partial
from _TTLoRAWrapper_TensorMultiplication import TTLoRALinearWrapper_withcores, AdaptCores_and_Test_Individual

# custom_cache_dir = "/lustre/vescratch1/ceodspspectrum/cache_hf/"
# os.environ['HF_DATASETS_CACHE'] = custom_cache_dir

def get_ttlora_shape(ttlora_shape_from_config):
    ttlora_shape = ttlora_shape_from_config
    return ttlora_shape

def get_ttlora_rank(r, ttlora_shape):
    ttlora_rank = [1]
    for i in range(len(ttlora_shape)-1):
        ttlora_rank.append(r)
    ttlora_rank.append(1)
    return ttlora_rank

def load_new_model_for_sequence_classification_from_local_path(config):
    model = AutoModelForSequenceClassification.from_pretrained(config["model_path"], num_labels=2)
    model.config.pad_token_id = model.config.eos_token_id
    for param in model.parameters():
        param.requires_grad = False
    # print(model)
    return model

def get_tokenizer(config, dataset):
    '''Tokenizes the provided dataset and data name using the tokenizer from the specified path'''
    path = config["tokenizer_path"]
    data_name = config["dataset_name"]

    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_text(batch):
        # Truncation true = truncate the tokenized text to max_length
        # Padding true = pad the tokenized text to max_length
        if data_name == "sst2":
            return tokenizer(batch["sentence"], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "mrpc":
            return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "cola":
            return tokenizer(batch["sentence"], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "qnli":
            return tokenizer(batch["question"], batch['sentence'], add_special_tokens=True, truncation=True, padding=True)
        # if data_name == "rte":
        #     return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding=True)
        # if data_name == "mnli":
        #     return tokenizer(batch["premise"], batch['hypothesis'], add_special_tokens=True, truncation=True, padding=True)
        # if data_name == "qqp":
        #     return tokenizer(batch["question1"], batch['question2'], add_special_tokens=True, truncation=True, padding=True)
        # if data_name == "stsb":
        #     return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding=True)
        # if data_name == "wsc":
        #     return tokenizer(batch["text"], batch['span1_text'], batch['span2_text'], add_special_tokens=True, truncation=True, padding=True)
        # if data_name == "winogrande":
        #     return tokenizer(batch["sentence"], batch['option1'], batch['option2'], batch['answer'], add_special_tokens=True, truncation=True, padding=True)
        # if data_name == "ax":
        #     return tokenizer(batch["premise"], batch['hypothesis'], add_special_tokens=True, truncation=True, padding=True)
        # if data_name == "multirc":
        #     return tokenizer(batch["paragraph"], batch['question'], batch['answer'], add_special_tokens=True, truncation=True, padding=True)
        # if data_name == "boolq":
        #     return tokenizer(batch["question"], batch['passage'], add_special_tokens=True, truncation=True, padding=True)
        # if data_name == "hellaswag":
        #     return tokenizer(batch["ind"], batch["activity_label"], batch['ctx_a'],batch['ctx_b'],batch['ctx'],batch['endings'],batch['source_id'],batch['split'],batch['split_type'], add_special_tokens=True, truncation=True, padding=True)
        
        #super_glue datasets
        if data_name == "cb":
            return tokenizer(batch["premise"], batch["hypothesis"], truncation=True, padding=True, max_length = 512)
        if data_name == "boolq":
            return tokenizer(batch["passage"], batch["question"], truncation=True, padding=True,max_length = 512)
        if data_name == "wsc":
            target = batch["target"]
            span1_text=[]
            span2_text=[]

            for entry in target:
                span1_text_val=entry.get('span1_text')
                span2_text_val=entry.get('span2_text')

                span1_text.append(span1_text_val)
                span2_text.append(span2_text_val)
            span = [f"{span1_text}{tokenizer.bos_token}{span2_text}" for span1_text, span2_text in zip(span1_text, span2_text)]
            return tokenizer(batch["text"], span, truncation=True, padding=True,max_length = 1024)
        if data_name == "copa":
            question = batch["question"]
            choice1 = batch["choice1"]
            choice2 = batch["choice2"]
            choice2 = batch["question"]
            combined = [f"{choice1}{tokenizer.bos_token}{choice2}{tokenizer.bos_token}{question}" for choice1, choice2, question in zip(choice1, choice2, question)]
            return tokenizer(batch["premise"], combined, truncation=True, padding=True,max_length = 1024)
    # Map the words in the dataset to the token values of the loaded tokenizer
    # None batch size = process entire dataset as single batch
    tokenized = dataset.map(tokenize_text, batched=True, batch_size=None) 

    ### change the format into tensors of the specific columns
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized

def get_mix_tokenizer(path, data_name, dataset): #used to do the maximum padding for the dataset to match all the types of dataset's sequence_length
    '''Tokenizes the provided dataset and data name using the tokenizer from the specified path'''
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_text(batch):
        if data_name == "sst2":
            return tokenizer(batch["sentence"], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        if data_name == "mrpc":
            return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        if data_name == "cola":
            return tokenizer(batch["sentence"], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        if data_name == "qnli":
            return tokenizer(batch["question"], batch['sentence'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        # if data_name == "rte":
        #     return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        # if data_name == "mnli":
        #     return tokenizer(batch["premise"], batch['hypothesis'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        # if data_name == "qqp":
        #     return tokenizer(batch["question1"], batch['question2'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        # if data_name == "stsb":
        #     return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        # if data_name == "wsc":
        #     return tokenizer(batch["text"], batch['span1_text'], batch['span2_text'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        # if data_name == "winogrande":
        #     return tokenizer(batch["sentence"], batch['option1'], batch['option2'], batch['answer'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        # if data_name == "ax":
        #     return tokenizer(batch["premise"], batch['hypothesis'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        # if data_name == "multirc":
        #     return tokenizer(batch["paragraph"], batch['question'], batch['answer'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        # if data_name == "boolq":
        #     return tokenizer(batch["question"], batch['passage'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        # if data_name == "hellaswag":
        #     return tokenizer(batch["ind"], batch["activity_label"], batch['ctx_a'],batch['ctx_b'],batch['ctx'],batch['endings'],batch['source_id'],batch['split'],batch['split_type'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        if data_name == "cb":
            return tokenizer(batch["premise"], batch["hypothesis"], truncation=True, padding='max_length', max_length=1024)
        if data_name == "boolq":
            return tokenizer(batch["passage"], batch["question"], truncation=True, padding='max_length', max_length = 1024)
        if data_name == "wsc":
            target = batch["target"]
            span1_text=[]
            span2_text=[]

            for entry in target:
                span1_text_val=entry.get('span1_text')
                span2_text_val=entry.get('span2_text')

                span1_text.append(span1_text_val)
                span2_text.append(span2_text_val)
            span = [f"{span1_text}{tokenizer.bos_token}{span2_text}" for span1_text, span2_text in zip(span1_text, span2_text)]
            return tokenizer(batch["text"], span, truncation=True, padding='max_length', max_length = 1024)
        if data_name == "copa":
            question = batch["question"]
            choice1 = batch["choice1"]
            choice2 = batch["choice2"]
            choice2 = batch["question"]
            combined = [f"{choice1}{tokenizer.bos_token}{choice2}{tokenizer.bos_token}{question}" for choice1, choice2, question in zip(choice1, choice2, question)]
            return tokenizer(batch["premise"], combined, truncation=True, padding='max_length', max_length = 1024)
    tokenized = dataset.map(tokenize_text, batched=True, batch_size=None) 
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized

def load_mixed_datasets(dataset_names, tokenizer_path):
    '''Dataset loading and check if loaded correctly'''
    mixed_train_dataset_dict = {
        
        "input_ids": torch.empty(0,dtype=torch.int64),
        "attention_mask": torch.empty(0, dtype=torch.int64),
        "label": torch.empty(0, dtype=torch.int64),
    }
    mixed_validation_dataset_dict = {
        "input_ids": torch.empty(0, dtype=torch.int64),
        "attention_mask": torch.empty(0, dtype=torch.int64),
        "label": torch.empty(0, dtype=torch.int64),
    }
    for dataset_name in dataset_names:
        take_train = 3668
        take_val = 408
        print("Loading dataset: ", dataset_name)
        dataset = load_dataset("glue",dataset_name)
        tokenized = get_mix_tokenizer(tokenizer_path, dataset_name , dataset)
        train_tokenized_dataset = tokenized["train"]
        train_tokenized_dataset = train_tokenized_dataset.remove_columns(
            [col for col in train_tokenized_dataset.column_names if col not in ["input_ids", "attention_mask", "label"]]
        )
        print("Train tokenized dataset: ", train_tokenized_dataset['input_ids'].shape, train_tokenized_dataset['attention_mask'].shape, train_tokenized_dataset['label'].shape)

        # print("Train tokenized dataset: ", train_tokenized_dataset['input_ids'].shape, train_tokenized_dataset['attention_mask'].shape, train_tokenized_dataset['label'].shape)
        validation_tokenized_dataset = tokenized["validation"]
        validation_tokenized_dataset = validation_tokenized_dataset.remove_columns(
            [col for col in train_tokenized_dataset.column_names if col not in ["input_ids", "attention_mask", "label"]]
        )
        print("Validation tokenized dataset: ", validation_tokenized_dataset['input_ids'].shape, validation_tokenized_dataset['attention_mask'].shape, validation_tokenized_dataset['label'].shape)

        #########################################For Train###################################################
        mixed_train_dataset_dict["input_ids"] = torch.cat((mixed_train_dataset_dict["input_ids"], 
                                                           train_tokenized_dataset["input_ids"][:take_train]), dim=0)
        mixed_train_dataset_dict["attention_mask"] = torch.cat((mixed_train_dataset_dict["attention_mask"], 
                                                                train_tokenized_dataset["attention_mask"][:take_train]), dim=0)
        mixed_train_dataset_dict["label"] = torch.cat((mixed_train_dataset_dict["label"], 
                                                       train_tokenized_dataset["label"][:take_train]), dim=0)
        #########################################For Validation###################################################

        mixed_validation_dataset_dict["input_ids"] = torch.cat((mixed_validation_dataset_dict["input_ids"], 
                                                                validation_tokenized_dataset["input_ids"][:take_val]), dim=0)
        mixed_validation_dataset_dict["attention_mask"] = torch.cat((mixed_validation_dataset_dict["attention_mask"], 
                                                                     validation_tokenized_dataset["attention_mask"][:take_val]), dim=0)
        mixed_validation_dataset_dict["label"] = torch.cat((mixed_validation_dataset_dict["label"], 
                                                            validation_tokenized_dataset["label"][:take_val]), dim=0)
    
    # print("mixed_train_dataset_dict: ", 
    #       mixed_train_dataset_dict['input_ids'].shape, 
    #       mixed_train_dataset_dict['attention_mask'].shape, 
    #       mixed_train_dataset_dict['label'].shape,
    #       "Data types: ", 
    #       mixed_train_dataset_dict['input_ids'].dtype, 
    #       mixed_train_dataset_dict['attention_mask'].dtype, 
    #       mixed_train_dataset_dict['label'].dtype
    #       )
    # print("mixed_validation_dataset_dict: ", 
    #       mixed_validation_dataset_dict['input_ids'].shape, 
    #       mixed_validation_dataset_dict['attention_mask'].shape, 
    #       mixed_validation_dataset_dict['label'].shape,
    #       "Data types: ", 
    #       mixed_validation_dataset_dict['input_ids'].dtype, 
    #       mixed_validation_dataset_dict['attention_mask'].dtype, 
    #       mixed_validation_dataset_dict['label'].dtype
    #       )
    # print(mixed_train_dataset_dict['label'].shape, mixed_train_dataset_dict['label'])
    # Shuffle the training dataset
    train_indices = torch.randperm(mixed_train_dataset_dict["input_ids"].size(0))
    mixed_train_dataset_dict["input_ids"] = mixed_train_dataset_dict["input_ids"][train_indices]
    mixed_train_dataset_dict["attention_mask"] = mixed_train_dataset_dict["attention_mask"][train_indices]
    mixed_train_dataset_dict["label"] = mixed_train_dataset_dict["label"][train_indices]

    # Shuffle the validation dataset
    val_indices = torch.randperm(mixed_validation_dataset_dict["input_ids"].size(0))
    mixed_validation_dataset_dict["input_ids"] = mixed_validation_dataset_dict["input_ids"][val_indices]
    mixed_validation_dataset_dict["attention_mask"] = mixed_validation_dataset_dict["attention_mask"][val_indices]
    mixed_validation_dataset_dict["label"] = mixed_validation_dataset_dict["label"][val_indices]
    
    return mixed_train_dataset_dict, mixed_validation_dataset_dict

def wrap_model_with_ttcores(model, config):
    
    ttlora_shape_q = get_ttlora_shape(config["qshape"])
    ttlora_rank_q = get_ttlora_rank(config["rank"], ttlora_shape_q)
    ttlora_shape_v = get_ttlora_shape(config["vshape"])
    ttlora_rank_v = get_ttlora_rank(config["rank"], ttlora_shape_v)

    m_factors_q = config["m_factors_q"]
    n_factors_q = config["n_factors_q"]
    m_factors_v = config["m_factors_v"]
    n_factors_v = config["n_factors_v"]

    ttlora_alpha = config["alpha"]
    ttlora_adapter_at_query = True
    ttlora_adapter_at_value = True

    assign_ttlora = partial(TTLoRALinearWrapper_withcores, alpha=ttlora_alpha)

    if "roberta" in config["model_name"]:
        for layer in model.roberta.encoder.layer:
            if ttlora_adapter_at_query:
                layer.attention.self.query = assign_ttlora(layer.attention.self.query,
                                                           tt_shape=ttlora_shape_q, 
                                                           tt_rank=ttlora_rank_q,
                                                           m_factors=m_factors_q,
                                                           n_factors=n_factors_q,
                                                           device=config["device"], 
                                                           init_choice=config["core_init_choice"]) 
            if ttlora_adapter_at_value:
                layer.attention.self.value = assign_ttlora(layer.attention.self.value,
                                                           tt_shape=ttlora_shape_v, 
                                                           tt_rank=ttlora_rank_v,
                                                           m_factors=m_factors_v,
                                                           n_factors=n_factors_v,
                                                           device=config["device"],
                                                           init_choice=config["core_init_choice"]) 
    elif "llama" in config["model_name"]:
        for layer in model.model.layers:
            if ttlora_adapter_at_query:
                layer.self_attn.q_proj = assign_ttlora(layer.self_attn.q_proj,
                                                       tt_shape=ttlora_shape_q, 
                                                       tt_rank=ttlora_rank_q,
                                                       m_factors=m_factors_q,
                                                       n_factors=n_factors_q,
                                                       device=config["device"],
                                                       init_choice=config["core_init_choice"])
            if ttlora_adapter_at_value:
                layer.self_attn.v_proj = assign_ttlora(layer.self_attn.v_proj,
                                                       tt_shape=ttlora_shape_v,
                                                       tt_rank=ttlora_rank_v,
                                                       m_factors=m_factors_v,
                                                       n_factors=n_factors_v,
                                                       device=config["device"],
                                                       init_choice=config["core_init_choice"])
    else:
        raise ValueError("Model name not recognized. Please use 'roberta' or 'llama' in the model name.")
    return model

def parse_experts(directory_path, model_name):

    """
    Parses all `.ckpt` files inside expert subfolders and organizes the experts into a nested dictionary.
    Saves ttlora cores and classifier weights for each expert.
    """
    # Nested dictionary to hold all experts
    all_experts = defaultdict(lambda: defaultdict(lambda: {"query": {}, "value": {}}))

    # Iterate through each expert folder in the directory
    for expert_name in os.listdir(directory_path):
        expert_folder = os.path.join(directory_path, expert_name)

        # Ensure it is a directory (expert folder)
        if os.path.isdir(expert_folder):
            # Iterate through .ckpt files inside the expert folder
            for filename in os.listdir(expert_folder):
                # Check if there are multiple .ckpt files in the expert folder
                ckpt_files = [f for f in os.listdir(expert_folder) if f.endswith(".ckpt")]
                if len(ckpt_files) > 1:
                    raise ValueError(f"Multiple .ckpt files found in {expert_folder}. Only one .ckpt file is allowed per expert folder.")
                if filename.endswith(".ckpt"):
                    file_path = os.path.join(expert_folder, filename)
                    # Load the .ckpt file
                    checkpoint = torch.load(file_path, map_location="cpu")
                    # Extract model weights (state_dict)
                    expert_data = checkpoint["state_dict"] 
                    if "roberta" in model_name: 
                        expert_data = {k: v for k, v in expert_data.items() if 'ttlora' in k or 'classifier' in k}
                        for full_key, tensor in expert_data.items():
                            tensor.requires_grad = False
                            parts = full_key.split(".")
                            if 'classifier' in parts:  #keys as: model.classifier.dense.weight
                                try:   
                                    classifier = parts[1]
                                    t_type = parts[2]
                                    w_b=parts[3]
                                    if t_type not in all_experts[expert_name][classifier]:
                                        all_experts[expert_name][classifier][t_type] = {}
                                    all_experts[expert_name][classifier][t_type][w_b] = tensor
                                except IndexError:
                                    print(f"Skipping invalid key: {full_key} in {expert_name}")  
                            else:  #keys: model.roberta.encoder.layer.11.attention.self.query.ttlora_cores.0
                                try:
                                    layer = f"layer_{parts[4]}"  # Extract layer index
                                    attention_type = parts[7]  # 'query' or 'value'
                                    ttlora_core = parts[-1]  # 'ttlora_cores.<index>'

                                    # Store extracted weights inside dictionary
                                    all_experts[expert_name][layer][attention_type][f'ttlora_core_{ttlora_core}'] = tensor
                                except IndexError:
                                    print(f"Skipping invalid key: {full_key} in {expert_name}")
                    
                    elif "llama" in model_name:
                        expert_data = {k: v for k, v in expert_data.items() if 'ttlora' in k or 'score' in k}
                        for full_key, tensor in expert_data.items():
                            tensor.requires_grad = False
                            parts = full_key.split(".")
                            if 'score' in parts:  #key as: model.score.weight (no bias)
                                try:   
                                    classifier = parts[1]
                                    w_b=parts[2]
                                    all_experts[expert_name][classifier][w_b] = tensor
                                except IndexError:
                                    print(f"Skipping invalid key: {full_key} in {expert_name}")  
                            else:        #model.model.layers.1.self_attn.q_proj.ttlora_cores.7
                                try:
                                    layer = f"layer_{parts[3]}"  # Extract layer index
                                    attention_type = parts[5]  # 'query' or 'value'
                                    if attention_type == "q_proj":
                                        attention_type = "query"
                                    elif attention_type == "v_proj":
                                        attention_type = "value"
                                    ttlora_core = parts[-1]  # 'ttlora_cores.<index>'

                                    # Store extracted weights inside dictionary
                                    all_experts[expert_name][layer][attention_type][f'ttlora_core_{ttlora_core}'] = tensor
                                except IndexError:
                                    print(f"Skipping invalid key: {full_key} in {expert_name}")

    return all_experts

# def load_dataset_from_local(config):
#     data_name = config["dataset_name"]
#     path = "./datasets/"

#     if data_name == "cb":
#         train_data_path = os.path.join(path, data_name.upper(), "train.jsonl")
#         val_data_path = os.path.join(path, data_name.upper(), "val.jsonl")

#         data_files={
#         "train": train_data_path,
#         "validation": val_data_path}

#         dataset = load_dataset("json", data_files = data_files)

#         unique_labels=dataset['train'].unique('label')
#         label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
#         def encode_labels(data):
#             data['label']=label_to_int[data['label']]
#             return data
#         encoded_data = dataset.map(encode_labels, batched=False)
#         return encoded_data

#     if data_name == "copa":
#         train_data_path = os.path.join(path, 'COPA', "train.jsonl")
#         val_data_path = os.path.join(path, 'COPA', "val.jsonl")

#         data_files={
#         "train": train_data_path,
#         "validation": val_data_path}

#         dataset = load_dataset("json", data_files = data_files)

#         return dataset

#     if data_name == "boolq":

#         def convert_labels(example):
#             example['label'] = 1 if example['answer'] else 0
#             # print(example)
#             return example
#         path_boolq = "/usr/projects/unsupgan/afia/stack_v2_boolq"
#         dataset = load_dataset(path_boolq)


#         dataset_int = dataset.map(convert_labels)
#         return dataset_int

#     if data_name == "wsc":
#         train_data_path = os.path.join(path, data_name.upper(), "train.jsonl")
#         val_data_path = os.path.join(path, data_name.upper(), "val.jsonl")

#         data_files={
#         "train": train_data_path,
#         "validation": val_data_path}

#         dataset = load_dataset("json", data_files = data_files)

#         return dataset

#     # dataset = load_dataset(data_path)
#     # dataset = load_dataset("glue", data_name)
# def convert_labels_wsc(example):
#     data=example.to_dict()
#     # del data['label']
#     data['label'] = [int(label) for label in example['label']]
#     print(type(data['label']))
#     return Dataset.from_dict(data)