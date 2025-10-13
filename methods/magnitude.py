import os
import time
import json
from tqdm import tqdm

import torch

from utils.data_utils import get_trainloaders
from utils.eval_utils import load_and_eval_ppl, eval_zero_shot
from utils.util import *


def get_layer_importance(modelhander, dataloader, bs=1, device=None, norm_power=1, weight_reduction="mean", block_reduction="mean"):
    if device == None:
        device = modelhander.model.device
    # Get input IDs
    testenc = dataloader.input_ids
    testattn = dataloader.attention_mask

    # Calculate number of samples
    nsamples = testenc.numel() // modelhander.model.seqlen
    
    layer_importance =  [0 for _ in range(modelhander.model.config.num_hidden_layers)]

    # Loop through each batch
    for i in tqdm(range(0, nsamples, bs), desc="Magnitude Score calculating ..."):
        # Calculate end index
        j = min(i+bs, nsamples)
        # Prepare inputs and move to device
        inputs = testenc[:, (i * modelhander.model.seqlen):(j * modelhander.model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, modelhander.model.seqlen)

        attn = testattn[:, (i * modelhander.model.seqlen):(j * modelhander.model.seqlen)].to(modelhander.model.device)
        attn = attn.reshape(j-i, modelhander.model.seqlen)

        # Forward pass through the model
        loss = modelhander.model(inputs, attention_mask=attn, labels=inputs).loss
        loss.backward()

        for idx, layer in enumerate(modelhander.layers):
            block_info = []
            for name, param in layer.named_parameters():
                if "proj" in name and "weight" in name:
                    weight_imp = param.detach().abs().pow(norm_power).sum(1)
                elif "norm" in name and "weight" in name:
                    weight_imp = param.detach().abs().pow(norm_power)
                else:
                    continue
                
                weight_imp = getattr(weight_imp, weight_reduction)(dim=0)
                weight_imp = weight_imp.item()
                block_info.append(weight_imp)

            block_imp = torch.tensor(block_info)
            block_imp = getattr(block_imp, block_reduction)(dim=0)
            block_imp = block_imp.item()

            layer_importance[idx] += block_imp

    layer_importance_average = [i/len(layer_importance) for i in layer_importance]

    return layer_importance_average


def main_func(args, modelhander):

    dataloader = get_trainloaders(args.calibration_dataset,
                                  tokenizer=modelhander.tokenizer,
                                  nsamples=args.nsamples,
                                  seed=args.seed,
                                  seqlen=modelhander.model.seqlen
                                  )

    magnitude_info = {}
    logging.info(f"Dataloader({args.calibration_dataset}) loaded.")
    logging.info("Compute the BI importance score of layers.")
    start_time = time.time()
    layer_importance = get_layer_importance(modelhander=modelhander, dataloader=dataloader, bs=1, device=modelhander.model.device, weight_reduction=args.weight_reduction, block_reduction=args.block_reduction)
    total_time = time.time() - start_time
    logging.info(f"Total time elapsed  (s): {total_time}")
    sorted_indices = sorted(range(len(layer_importance)), key=lambda i: layer_importance[i], reverse=False)
    
    if args.heuristic:
        layer_idxs = list(range(modelhander.config.num_hidden_layers))
        heuristic_list = layer_idxs[:4] + layer_idxs[-2:]
        for idx in heuristic_list:
            sorted_indices.remove(idx)
        sorted_indices.extend(heuristic_list)

    logging.info(f"The BI score of Layers:")
    for idx, score in zip(list(range(modelhander.config.num_hidden_layers)), layer_importance):
        logging.info(f"{idx:>{3}} → {score}")

    logging.info(f"Remove order of layer index: \n{sorted_indices}")

    magnitude_info["layer_importance"] = layer_importance
    magnitude_info["sorted_indices"] = sorted_indices
    magnitude_info["time_elapsed"] = total_time

    num_remove_blocks = modelhander.config.num_hidden_layers - args.target_layers
    if args.continue_saving:
        logging.info("Continues saving sub-model")
        removal_list = []
        for idx in sorted_indices[:num_remove_blocks]:
            removal_list.append(idx)
            logging.info("*" * 10, f"Remove layer index is {idx}", "*" * 10)
            cut_num = len([x for x in removal_list if x < idx])
            idx = idx - cut_num
            modelhander.remove_layers(removal_list=idx)
            save_path = os.path.join(args.save_path, args.save_name + f"_{modelhander.model.config.num_hidden_layers}")
            modelhander.save(path=save_path)
            magnitude_info[f"Layers_{modelhander.model.config.num_hidden_layers}"] = {"remove_layer_idx_list": copy.copy(removal_list)}
            logging.info(f"Save model to {save_path}")
            for da in args.ppl_data:
                logging.info(f"Starting {da} PPL evaluation...")
                ppl = load_and_eval_ppl(modelhander.model, dataset=da, tokenizer=modelhander.tokenizer)
                magnitude_info[f"Layers_{modelhander.model.config.num_hidden_layers}"][f"{da}_ppl"] = ppl

    else:
        removal_list=sorted_indices[:num_remove_blocks]
        logging.info(f"Remove layer index list is {removal_list}")
        modelhander.remove_layers(removal_list=removal_list)
        save_path = os.path.join(args.save_path, args.save_name + f"_{modelhander.model.config.num_hidden_layers}")
        modelhander.save(path=save_path)
        logging.info(f"Save model to {save_path}")
        magnitude_info[f"Layers_{modelhander.model.config.num_hidden_layers}"] = {"remove_layer_idx_list": removal_list}
        for da in args.ppl_data:
            logging.info(f"Starting {da} PPL evaluation...")
            ppl = load_and_eval_ppl(modelhander.model, dataset=da, tokenizer=modelhander.tokenizer)
            magnitude_info[f"Layers_{modelhander.model.config.num_hidden_layers}"][f"{da}_ppl"] = ppl

    logging.info(f"Save {args.method} information to {os.path.join(args.save_path, f'{args.method}_info.json')}")
    with open(os.path.join(args.save_path, f'{args.method}_info.json'), 'w') as json_file:
        json.dump(magnitude_info, json_file, indent=4)