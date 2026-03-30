import os
import random

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer

DATA_DIR = "/workspace/wangfei154/datasets"
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

def get_tokenizer(model):
    if "llama" in model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
        # fix for transformer 4.28.0.dev0 compatibility
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    return tokenizer

def get_wikitext2(nsamples, seed, seqlen, tokenizer, batch_size):
    
    traindata = load_dataset(os.path.join(DATA_DIR, 'wikitext'), 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset(os.path.join(DATA_DIR, 'wikitext'), 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    new_trainloader = []
    num_batches = nsamples // batch_size + (int)(nsamples % batch_size > 0)
    for i in range(0, num_batches):
        start =  i * batch_size
        end = min(start + batch_size, nsamples)
        batched_inp = []
        batched_tar = []
        for j in range(start, end):
            batched_inp.append(trainloader[j][0])
            batched_tar.append(trainloader[j][1])
        batched_inp = torch.cat(batched_inp)
        batched_tar = torch.cat(batched_tar)
        new_trainloader.append((batched_inp, batched_tar))
    del trainloader
    trainloader = new_trainloader
    del new_trainloader

    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, tokenizer, batch_size):
   
    traindata = load_dataset(
        os.path.join(DATA_DIR, 'allenai/c4'), data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(os.path.join(DATA_DIR, 'allenai/c4'), data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    new_trainloader = []
    num_batches = nsamples // batch_size + (int)(nsamples % batch_size > 0)
    for i in range(0, num_batches):
        start =  i * batch_size
        end = min(start + batch_size, nsamples)
        batched_inp = []
        batched_tar = []
        for j in range(start, end):
            batched_inp.append(trainloader[j][0])
            batched_tar.append(trainloader[j][1])
        batched_inp = torch.cat(batched_inp)
        batched_tar = torch.cat(batched_tar)
        new_trainloader.append((batched_inp, batched_tar))
    del trainloader
    trainloader = new_trainloader
    del new_trainloader

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None, batch_size=1):
    if 'wiki2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer, batch_size)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, tokenizer, batch_size)

def get_wikitext2_trainenc(seed, nsamples, tokenizer):
    
    traindata = load_dataset(os.path.join(DATA_DIR, 'wikitext'), 'wikitext-2-raw-v1', split='train')
    traindata = traindata.shuffle(seed=seed)
    trainenc = tokenizer("\n\n".join(traindata[:nsamples]['text']), return_tensors='pt')

    return trainenc

def get_c4_trainenc(seed, nsamples, tokenizer):
    traindata = load_dataset(
        os.path.join(DATA_DIR, 'allenai/c4'), data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(os.path.join(DATA_DIR, 'allenai/c4'), data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    traindata = traindata.shuffle(seed=seed)
    
    trainenc = tokenizer(' '.join(traindata[:nsamples]['text']), return_tensors='pt')

    return trainenc

def get_pg19_bookcorpus_trainenc(seed, nsamples, tokenizer, dataset="pg19"):
    
    traindata = load_dataset(os.path.join(DATA_DIR, dataset), split='train', trust_remote_code=True)
    traindata = traindata.shuffle(seed=seed)
    trainenc = tokenizer("\n\n".join(traindata[:nsamples]['text']), return_tensors='pt')

    return trainenc

def get_alpaca_trainenc(seed, nsamples, tokenizer, seqlen=2048):
    traindata = load_dataset(os.path.join(DATA_DIR, "alpaca-cleaned"), split='train')
    traindata = traindata.shuffle(seed=seed)
    data = ["\n".join([i, j, k]) for i, j, k in zip(traindata[:nsamples]['instruction'], traindata[:nsamples]['input'], traindata[:nsamples]['output'])]
    trainenc = tokenizer(data, return_tensors='pt', max_length=seqlen, padding='max_length')
    trainenc["input_ids"] = trainenc["input_ids"].reshape(1, -1)
    trainenc["attention_mask"] = trainenc["attention_mask"].reshape(1, -1)
    return trainenc


def get_mmlu_trainenc(seed, nsamples, tokenizer, seqlen=2048, num_tasks=None):
    from datasets import Dataset
    from tqdm import tqdm
    subclass = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    if num_tasks != None:
        random.shuffle(subclass)
        subclass = subclass[:num_tasks]
    keys = ["question", "subject", "choices", "answer"]
    subnum = nsamples // len(subclass)
    extra = nsamples % len(subclass)
    num_list = [subnum+1] * extra + [subnum] * (len(subclass)-extra)

    random.shuffle(num_list)

    traindata = {key:[] for key in keys}

    for num, classsname in tqdm(zip(num_list, subclass), total=len(subclass), desc="Loading the subclass in MMLU"):
        try:
            data = load_dataset(os.path.join(DATA_DIR, "mmlu"), classsname, split="train").shuffle(seed=seed)[:num]
        except:
            data = load_dataset(os.path.join(DATA_DIR, "mmlu"), classsname, split="validation").shuffle(seed=seed)[:num]
        for key in keys:
            traindata[key].extend(data[key])

    dataset = Dataset.from_dict(traindata)

    all_data = format_mmlu_example(dataset, include_answer=False)

    trainenc = tokenizer(all_data, return_tensors='pt', max_length=seqlen, padding='max_length', truncation=True)
    trainenc["input_ids"] = trainenc["input_ids"].reshape(1, -1)
    trainenc["attention_mask"] = trainenc["attention_mask"].reshape(1, -1)
    return trainenc

def format_mmlu_example(dataset, include_answer=True):
    # Define the possible choices for multiple-choice questions
    choices = ["A", "B", "C", "D"]
    
    data = []
    for ques, cho, ans in zip(dataset["question"], dataset["choices"], dataset["answer"]):
        prompt = ques
        for i in range(len(cho)):
            prompt += "\n{}. {}".format(choices[i], cho[i])
        prompt += "\nAnswer:"
        
        if include_answer:
            prompt += " {}. {}\n\n".format(choices[ans], choices[ans])

        data.append(prompt)
  
    return data


def get_trainloaders(name, tokenizer, nsamples=128, seed=0, seqlen=2048, batch_size=1, num_tasks=None):
    if 'wiki2' in name:
        return get_wikitext2_trainenc(seed, nsamples, tokenizer)
    if 'c4' in name:
        return get_c4_trainenc(seed, nsamples, tokenizer)
    if 'pg19' in name:
        return get_pg19_bookcorpus_trainenc(seed, nsamples, tokenizer, dataset="pg19")
    if 'bookcorpus' in name:
        return get_pg19_bookcorpus_trainenc(seed, nsamples, tokenizer, dataset="bookcorpus")
    if 'alpaca' in name:
        return get_alpaca_trainenc(seed, nsamples, tokenizer, seqlen=seqlen)
    if 'mmlu' in name:
        return get_mmlu_trainenc(seed, nsamples, tokenizer, seqlen=seqlen, num_tasks=num_tasks)
    raise NameError(f"{name} is not implemented.")