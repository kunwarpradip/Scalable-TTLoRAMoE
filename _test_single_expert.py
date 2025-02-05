import pytorch_lightning as pl
import pandas as pd
import torch
import tensorly as tl
import time
import os
from tqdm import tqdm
import copy
import warnings
import sys

from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from functools import partial
from datasets import load_dataset
from model import CustomLightningModule
from utils import get_tokenizer, get_ttlora_shape, get_ttlora_rank, parse_experts
from utils import load_new_model_for_sequence_classification_from_local_path, wrap_model_with_ttcores
from _TTLoRAWrapper_TensorMultiplication import AdaptCores_and_Test_Individual, TTLoRALinearWrapper_withcores


tl.set_backend('pytorch')


# Suppress all warnings
warnings.filterwarnings("ignore")

sys.stdout = open('output.log', 'w')
sys.stderr = open('output.log', 'w')

def load_best_lightning_model_from_checkpoint(model, config):
    print("\n","*"*50,
          f"Loading the best lightning model for {config["dataset_name"]} from the checkpoint",
          "*"*50,"\n")
    best_path = config["expert_path"][config["dataset_name"]]
    loaded_lightning_model = CustomLightningModule.load_from_checkpoint(
        best_path, 
        model=model, 
        config=config)
    return loaded_lightning_model

def evaluate_lightning_model_and_print(trainer, lightning_model, train_loader, val_loader):
    # traindata_acc = trainer.test(lightning_model, dataloaders=train_loader, verbose=False)
    validata_acc=trainer.test(lightning_model, dataloaders=val_loader,verbose=False)
    print(
        # "Training Data Accuracy: \n", traindata_acc, 
          "\nValidation Data Accuracy: \n", validata_acc,)

def adapt_classifier_with_expert_weightandbias(model, config):
    if "roberta" in config["model_name"]:
        #consistency in classifier weights as it changes every time we load a new roberta model for sequence classification
        model.classifier.dense.weight.data = config["experts_dict"][config["dataset_name"]]["classifier"]["dense"]['weight'] 
        model.classifier.dense.bias.data = config["experts_dict"][config["dataset_name"]]["classifier"]["dense"]['bias']
        model.classifier.out_proj.weight.data = config["experts_dict"][config["dataset_name"]]["classifier"]["out_proj"]['weight']
        model.classifier.out_proj.bias.data =  config["experts_dict"][config["dataset_name"]]["classifier"]["out_proj"]['bias']
    
    elif "llama" in config["model_name"]:
        model.score.weight.data = config["experts_dict"][config["dataset_name"]]["score"]['weight']
        #no bias present in llama model
    
    else:
        raise ValueError("Model name not recognized. Please use 'roberta' or 'llama' in the model name.")

def adapt_query_and_value_with_expert_cores(model, config):
        ttlora_adapter_at_query = True
        ttlora_adapter_at_value = True
        experts = config["experts_dict"]
        if "roberta" in config["model_name"]:
            layer_idx = 0
            for layer in model.roberta.encoder.layer:
                if ttlora_adapter_at_query:
                    ttlora_cores_q = experts[config["dataset_name"]][f'layer_{layer_idx}']["query"]
                    stack_ttlora_cores_q = [core.to("cuda") for key, core in ttlora_cores_q.items()]
                    layer.attention.self.query = AdaptCores_and_Test_Individual(module = layer.attention.self.query,
                                                                alpha=config["alpha"],
                                                                m_factors=config["m_factors_q"],
                                                                n_factors=config["n_factors_q"],
                                                                layer_idx=layer_idx, 
                                                                ttlora_cores=stack_ttlora_cores_q,
                                                                device = config["device"])
                if ttlora_adapter_at_value:
                    ttlora_cores_v = experts[config["dataset_name"]][f'layer_{layer_idx}']["value"]
                    stack_ttlora_cores_v = [core.to("cuda") for key, core in ttlora_cores_v.items()]
                    layer.attention.self.value = AdaptCores_and_Test_Individual(module = layer.attention.self.value,
                                                                alpha=config["alpha"],
                                                                m_factors=config["m_factors_v"],
                                                                n_factors=config["n_factors_v"],
                                                                layer_idx=layer_idx, 
                                                                ttlora_cores=stack_ttlora_cores_v,
                                                                device = config["device"])
                layer_idx += 1
        elif "llama" in config["model_name"]:
            layer_idx = 0
            for layer in model.model.layers:
                if ttlora_adapter_at_query:
                    ttlora_cores_q = experts[config["dataset_name"]][f'layer_{layer_idx}']["query"]
                    stack_ttlora_cores_q = [core.to("cuda") for key, core in ttlora_cores_q.items()]
                    layer.self_attn.q_proj = AdaptCores_and_Test_Individual(module = layer.self_attn.q_proj,
                                                                alpha=config["alpha"],
                                                                m_factors=config["m_factors_q"],
                                                                n_factors=config["n_factors_q"],
                                                                layer_idx=layer_idx, 
                                                                ttlora_cores=stack_ttlora_cores_q, 
                                                                device = config["device"])
                if ttlora_adapter_at_value:
                    ttlora_cores_v = experts[config["dataset_name"]][f'layer_{layer_idx}']["value"]
                    stack_ttlora_cores_v = [core.to("cuda") for key, core in ttlora_cores_v.items()]
                    layer.self_attn.v_proj = AdaptCores_and_Test_Individual(module = layer.self_attn.v_proj,
                                                                alpha=config["alpha"],
                                                                m_factors=config["m_factors_v"],
                                                                n_factors=config["n_factors_v"],
                                                                layer_idx=layer_idx, 
                                                                ttlora_cores=stack_ttlora_cores_v,
                                                                device = config["device"])
                layer_idx += 1
        else:
            raise ValueError("Model name not recognized. Please use 'roberta' or 'llama' in the model name.")
        
def train_without_ray(config):

    if not torch.cuda.is_available():
        print("Please switch to a GPU machine before running this notebook.")
    dataset = load_dataset(config["glue_type"], config["dataset_name"]) 
    
    tokenized = get_tokenizer(config, dataset)
    train_dataset = tokenized["train"]
    val_dataset = tokenized["validation"] 
    #we don't use test datasets as they contain hidden labels as -1

    '''Dataloader (an iterable) handles number of rows in each batch and how many gpus to use'''
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32, #256 for roberta, 512 for llama
        shuffle=True,   #data shuffles at the beginning of each epoch
        num_workers=8   #8 for roberta, 16 for llama
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=32, #256 for roberta, 512 for llama
        num_workers=8  #8 for roberta, 16 for llama
        #no need to shuffle the validation data as to get the consistent evaluations
    )

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        precision="16-mixed",
        devices=args.gpus,
        # log_every_n_steps=10,
        enable_progress_bar=False,
    )
    
    initial_model = load_new_model_for_sequence_classification_from_local_path(config)
    print("\n","*"*50,
          f"First Loaded Untrained {config["model_name"]} Summary","*"*50,"\n", 
          initial_model) 
    
    #create lightning model just to evaluate using the trainer from pytorch_lightning
    initial_lightning_model = CustomLightningModule(initial_model, config)
    print("\n","*"*50, f"Evaluations With Untrained {config["model_name"]}","*"*50,"\n")
    evaluate_lightning_model_and_print(trainer, initial_lightning_model, train_loader, val_loader)

    #Wrap the model with TTLoRA
    ttwrapped_untrained_model = wrap_model_with_ttcores(initial_model, config)
    ttwrapped_untrained_lightning_model = CustomLightningModule(ttwrapped_untrained_model, config)
    print("\n","*"*50,
          f"Model summary of Untrained TT-Wrapped Lightning {config["model_name"]}:","*"*50,"\n", 
          ttwrapped_untrained_lightning_model) 
    print("\n","*"*50, f"Evaluations With Untrained TTWrapped Lightning {config["model_name"]}","*"*50,"\n")
    evaluate_lightning_model_and_print(trainer, ttwrapped_untrained_lightning_model, train_loader, val_loader)
    
    #Load the best model from the checkpoint
    loaded_best_trained_lightning_model = load_best_lightning_model_from_checkpoint(ttwrapped_untrained_model, config)
    print("\n","*"*50,
          f"Model summary of checkpoint loaded best trained{config["model_name"]} TT-Wrapped Lightning:","*"*50 ,"\n", 
          loaded_best_trained_lightning_model)  
    
    print("\n","*"*50, f"Evaluations After Loading Best {config["model_name"]} TT-Wrapped Lightning Checkpoints","*"*50)
    evaluate_lightning_model_and_print(trainer, loaded_best_trained_lightning_model, train_loader, val_loader)

    new_model_for_adaptation = load_new_model_for_sequence_classification_from_local_path(config)
    adapt_classifier_with_expert_weightandbias(new_model_for_adaptation, config)
    adapt_query_and_value_with_expert_cores(new_model_for_adaptation, config)
    new_light = CustomLightningModule(new_model_for_adaptation, config)

    print("\n","*"*50, f"Evaluations After Adapting {config["model_name"]} with Loaded TT-Cores and Classifier From Expert {config["dataset_name"]}","*"*50)
    evaluate_lightning_model_and_print(trainer, new_light, train_loader, val_loader)   
    print(config["experts_dict"].keys())
    print(config["experts_dict"][config["dataset_name"]])
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    
    #changeable parameter
    '''while changing the model_name, make sure to change the expert_path as well'''
    model_name = "llama-3-8b" # roberta-base, llama-3-8b, llama-3-70b
    #changeable parameter
    if model_name == "roberta-base":
        expert_path= {
            "mrpc": f"./trained_checkpoints/{model_name}/experts/mrpc/{os.listdir(f'./trained_checkpoints/{model_name}/experts/mrpc/')[0]}",
            "cola": f"./trained_checkpoints/{model_name}/experts/cola/{os.listdir(f'./trained_checkpoints/{model_name}/experts/cola/')[0]}",
            "sst2": f"./trained_checkpoints/{model_name}/experts/sst2/{os.listdir(f'./trained_checkpoints/{model_name}/experts/sst2/')[0]}",
            "qnli": f"./trained_checkpoints/{model_name}/experts/qnli/{os.listdir(f'./trained_checkpoints/{model_name}/experts/qnli/')[0]}",
                    }
    elif "llama" in model_name:
        expert_path= {
            "cb": f"./trained_checkpoints/{model_name}/experts/cb/{os.listdir(f'./trained_checkpoints/{model_name}/experts/cb/')[0]}",
            
                    }
    else:
        raise ValueError("Model name not recognized. Please use 'roberta' or 'llama' in the model name.")

    experts_dict = parse_experts(f"./trained_checkpoints/{model_name}/experts", model_name)
    print(experts_dict.keys())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = {
        #ttlora parameters
        #query parameters
        "qshape": [64,4,3,3,4,64] if "roberta-base" in model_name #roberta query shape = 768x768
        else [16,4,4,4,2,2,2,2,4,4,4,16] if "llama-3-8b" in model_name #llama-3-8b q_proj shape = 4096x4096,
        else [16,4,4,4,2,2,2,2,2,2,4,4,4,16] if "llama-3-70b" in model_name #llama-3-70b q_proj shape = 8192x8192
        else ValueError(f"{model_name} Not adapted for this experiment"), 

        "m_factors_q": 
        [64,4,3] if "roberta-base" in model_name #roberta m of query shape = 768,
        else [16,4,4,4,2,2] if "llama-3-8b" in model_name #llama-3-8b m of q_proj shape = 4096,
        else [16,4,4,4,2,2,2] if "llama-3-70b" in model_name #llama-3-70b m of q_proj shape = 8192
        else ValueError(f"{model_name} Not adapted for this experiment"), 

        "n_factors_q": 
        [64,4,3] if "roberta-base" in model_name #roberta n of query shape = 768
        else [16,4,4,4,2,2] if "llama-3-8b" in model_name #llama-3-8b n of q_proj shape = 4096,
        else [16,4,4,4,2,2,2] if "llama-3-70b" in model_name #llama-3-70b n of q_proj shape = 8192
        else ValueError(f"{model_name} Not adapted for this experiment"),

        #value parameters
        "vshape": [64,4,3,3,4,64] if "roberta-base" in model_name #roberta value shape = 768x768
        else [16,4,4,4,2,2,2,2,4,4,16] if "llama-3-8b" in model_name #llama-3-8b v_proj shape = 4096x1024,
        else [16,4,4,4,2,2,2,2,2,4,4,16] if "llama-3-70b" in model_name #llama-3-70b v_proj shape = 8192x1024
        else ValueError(f"{model_name} Not adapted for this experiment"), 

        "m_factors_v": 
        [64,4,3] if "roberta-base" in model_name #roberta m of value shape = 768
        else [16,4,4,4,2,2] if "llama-3-8b" in model_name #llama-3-8b m of q_proj shape = 4096,
        else [16,4,4,4,2,2,2] if "llama-3-70b" in model_name #llama-3-70b m of q_proj shape = 8192
        else ValueError(f"{model_name} Not adapted for this experiment"), 

        "n_factors_v": 
        [64,4,3] if "roberta-base" in model_name #roberta n of value shape = 768
        else [16,4,4,2,2] if "llama-3-8b" in model_name #llama-3-8b n of q_proj shape = 1024
        else [16,4,4,2,2] if "llama-3-70b" in model_name #llama-3-70b n of q_proj shape = 1024
        else ValueError(f"{model_name} Not adapted for this experiment"),

        
        "rank": 
        4 if "roberta-base" in model_name 
        else 10 if "llama-3-8b" in model_name
        else 10 if "llama-3-70b" in model_name
        else ValueError(f"{model_name} Not adapted for this experiment"),

        "alpha": 
        8 if "roberta-base" in model_name 
        else 12 if "llama-3-8b" in model_name
        else 12 if "llama-3-70b" in model_name
        else ValueError(f"{model_name} Not adapted for this experiment"),
        
        "common alpha": 
        8 if "roberta-base" in model_name 
        else 12 if "llama-3-8b" in model_name
        else 12 if "llama-3-70b" in model_name
        else ValueError(f"{model_name} Not adapted for this experiment"),

        #expert parameters
        "experts_dict": experts_dict,
        "expert_path": expert_path,

        #model parameters
        "model_name" : model_name,
        "model_path" : f"./models/{model_name}/{model_name}-model",
        "tokenizer_path" : f"./models/{model_name}/{model_name}-tokenizer",
        "device": device,  

        #changable dataset parameters:
        "glue_type": "super_glue", # glue, super_glue
        "dataset_name" : "cb", # glue for roberta are :mrpc, cola, sst2, qnli, super_glue for llama are: boolq, cb, copa, wsc
        
        #changeable hyperparameters
        "learning_rate":  
        1e-1 if "roberta-base" in model_name 
        else 1e-1 if "llama-3-8b" in model_name
        else 1e-1 if "llama-3-70b" in model_name
        else ValueError(f"{model_name} Not adapted for this experiment"),
    }
    
    analysis =  train_without_ray(config)

