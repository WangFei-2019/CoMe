import copy
import fire
import os
from typing import List
import shutil

import torch

from utils.model_utils import get_llmhander
from utils.eval_utils import load_and_eval_ppl, eval_zero_shot
from utils.util import init_logging

CACHE_DIR = ".cache"
def eval(
        model_name: str = 'meta-llama/Llama-2-7b-hf',
        removal_list: List[int] = [],
        save_results: bool = False,
        result_folder: str = 'results',
        result_file: str = 'eval.txt',
        device: int = 0,
        eval_zeroshot: bool = False
    ):
    
    init_logging()
    modelheader = get_llmhander(model_name)
    logging.info(f"Loaded Model: {model_name}")
    modelheader.model.eval()
    
    original_removal_list = copy.deepcopy(removal_list)
    removal_list.sort()
    modelheader.remove_layers(removal_list=removal_list)
    
    logging.info(f"Starting PPL evaluation...")
    ppl_list = {}
    test_datasets = ['wiki2', 'c4']
    for dataset in test_datasets:
        ppl = load_and_eval_ppl(modelheader.model, device, dataset=dataset, tokenizer=modelheader.tokenizer)

        logging.info(f"{dataset} perplexity = {ppl:.2f}")

        ppl_list[dataset] = ppl
    

    modelheader.save(CACHE_DIR)
    del modelheader
    torch.cuda.empty_cache()

    if eval_zeroshot:
        logging.info(f"Starting Zero-shot tasks evaluation...")
        if '30b' or '66b' or '70b' in model_name:
            parallelize = True
        else:
            parallelize = False

        tasks = ['piqa','winogrande','hellaswag','arc_challenge','arc_easy']

        results = eval_zero_shot(CACHE_DIR, tasks, parallelize=parallelize)
        results = results['results']
        shutil.rmtree(CACHE_DIR)


        for task in tasks:
            logging.info(f"{task}: {results[task]}")

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    result_path = os.path.join(result_folder, result_file)
    
    if save_results:
        with open(result_path, 'a') as file:
            sentences = []
            sentences.append(f"Model Name: {model_name}\n")
            sentences.append(f"Block Removal Order: {original_removal_list}\n")
            
            if eval_zeroshot:
                sentences.append(f"WikiText-2 PPL: {ppl_list['wikitext2']:.2f}\n")
                sentences.append(f"C4 PPL: {ppl_list['c4']:.2f}\n")
                sentences.append(f"Zero-shot results: \n")
                for task in tasks:
                    sentences.append(f"{task}: {results[task]}\n")
                sentences.append("\n")
            else:
                sentences.append(f"WikiText-2 PPL: {ppl_list['wikitext2']:.2f} ")
                sentences.append(f"C4 PPL: {ppl_list['c4']:.2f}\n\n")
            
            for sentence in sentences:
                file.write(sentence)

if __name__ == "__main__":
    fire.Fire(eval)