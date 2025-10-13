import os
import time
import json
from tqdm import tqdm

import torch

from utils.data_utils import get_trainloaders
from utils.eval_utils import load_and_eval_ppl, eval_zero_shot
from utils.util import *

def block_influence(
    input_hidden_state: torch.Tensor,
    output_hidden_state: torch.Tensor,
    angular=False,
):
    """
    input_hidden_state: B, S, D
    output_hidden_state: B, S, D
    """
    _, _, d = input_hidden_state.shape
    input_hidden_state = input_hidden_state.reshape(-1, d)
    output_hidden_state = output_hidden_state.reshape(-1, d)

    norm_input = input_hidden_state.norm(dim=-1, keepdim=True)
    norm_output = output_hidden_state.norm(dim=-1, keepdim=True)

    sim = (input_hidden_state @ output_hidden_state.T) / (norm_input * norm_output)
    sim = sim.diagonal().nan_to_num(nan=0.5)

    sim = torch.clamp(sim, min=0, max=1)

    if angular:
        return (torch.arccos(sim) / torch.pi)

    # assert ((1 - sim) <= 1).any().item()
    return 1 - sim


def compute_bi(hiddens: List[torch.Tensor], angular: bool=False, n_prune_layers=None):
    importances = [0 for _ in range(len(hiddens)-1)]  # layer-wise importance scores
    n = 1
    if angular:
        assert n_prune_layers is not None, "Set number of layers to prune to use angular importance"
        n = n_prune_layers

    for i in range(len(hiddens) - n):
        in_hidden = hiddens[i]
        out_hidden = hiddens[i+n]
        if angular:
            # use only last token for angular distance as described in section 3.2
            # https://arxiv.org/pdf/2403.17887.pdf
            in_hidden = in_hidden[:,-1:]
            out_hidden = out_hidden[:,-1:]
        
        importances[i] += block_influence(
            in_hidden,
            out_hidden,
            angular=angular
        ).sum().cpu().item()
    
    return importances

@torch.inference_mode()
def get_layer_importance(model, dataloader, bs=1, device=None, angular=False):
    if device == None:
        device = model.device
    # Get input IDs
    testenc = dataloader.input_ids
    testattn = dataloader.attention_mask

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen
    
    layer_importance = [0 for _ in range(model.config.num_hidden_layers)]

    # Loop through each batch
    for i in tqdm(range(0, nsamples, bs), desc="BI Score calculating ..."):
        # Calculate end index
        j = min(i+bs, nsamples)
        # Prepare inputs and move to device
        inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        attn = testattn[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        attn = attn.reshape(j-i, model.seqlen)

        # Forward pass through the model
        hidden_states = model(inputs, attention_mask=attn, output_hidden_states=True).hidden_states

        layer_importance = list_item_sum(compute_bi(hidden_states, angular=angular), layer_importance)

    layer_importance_average = [i/len(layer_importance) for i in layer_importance]

    return layer_importance_average


def main_func(args, modelhander):

    dataloader = get_trainloaders(args.calibration_dataset,
                                  tokenizer=modelhander.tokenizer,
                                  nsamples=args.nsamples,
                                  seed=args.seed,
                                  seqlen=modelhander.model.seqlen
                                  )

    shortgpt_info = {}
    logging.info(f"Dataloader({args.calibration_dataset}) loaded.")
    logging.info("Compute the BI importance score of layers.")
    start_time = time.time()
    layer_importance = get_layer_importance(model=modelhander.model, dataloader=dataloader, bs=1, device=modelhander.model.device, angular=False)
    total_time = time.time() - start_time
    logging.info(f"Total time elapsed  (s): {total_time}")
    sorted_indices = sorted(range(len(layer_importance)), key=lambda i: layer_importance[i], reverse=False)
    logging.info(f"The BI score of Layers:")
    for idx, score in zip(list(range(modelhander.config.num_hidden_layers)), layer_importance):
        logging.info(f"{idx:>{3}} → {score}")

    logging.info(f"Remove order of layer index: \n{sorted_indices}")

    shortgpt_info["layer_importance"] = layer_importance
    shortgpt_info["sorted_indices"] = sorted_indices
    shortgpt_info["time_elapsed"] = total_time

    num_remove_blocks = modelhander.config.num_hidden_layers - args.target_layers
    if args.continue_saving:
        logging.info("Continues saving sub-model")
        removal_list = []
        for idx in sorted_indices[:num_remove_blocks]:
            removal_list.append(idx)
            logging.info("*" * 10, f"Layers {modelhander.model.config.num_hidden_layers} → {modelhander.model.config.num_hidden_layers-1}. Remove layer index is {idx}", "*" * 10)
            cut_num = len([x for x in removal_list if x < idx])
            idx = idx - cut_num
            modelhander.remove_layers(removal_list=idx)
            save_path = os.path.join(args.save_path, args.save_name + f"_{modelhander.model.config.num_hidden_layers}")
            modelhander.save(path=save_path)
            shortgpt_info[f"Layers_{modelhander.model.config.num_hidden_layers}"] = {"remove_layer_idx_list": copy.copy(removal_list)}
            logging.info(f"Save model to {save_path}")
            for da in args.ppl_data:
                logging.info(f"Starting {da} PPL evaluation...")
                ppl = load_and_eval_ppl(modelhander.model, dataset=da, tokenizer=modelhander.tokenizer)
                shortgpt_info[f"Layers_{modelhander.model.config.num_hidden_layers}"][f"{da}_ppl"] = ppl

    else:
        removal_list=sorted_indices[:num_remove_blocks]
        logging.info(f"Remove layer index list is {removal_list}")
        modelhander.remove_layers(removal_list=removal_list)
        save_path = os.path.join(args.save_path, args.save_name + f"_{modelhander.model.config.num_hidden_layers}")
        modelhander.save(path=save_path)
        logging.info(f"Save model to {save_path}")
        shortgpt_info[f"Layers_{modelhander.model.config.num_hidden_layers}"] = {"remove_layer_idx_list": removal_list}
        for da in args.ppl_data:
            logging.info(f"Starting {da} PPL evaluation...")
            ppl = load_and_eval_ppl(modelhander.model, dataset=da, tokenizer=modelhander.tokenizer)
            shortgpt_info[f"Layers_{modelhander.model.config.num_hidden_layers}"][f"{da}_ppl"] = ppl

    logging.info(f"Save {args.method} information to {os.path.join(args.save_path, f'{args.method}_info.json')}")
    with open(os.path.join(args.save_path, f'{args.method}_info.json'), 'w') as json_file:
        json.dump(shortgpt_info, json_file, indent=4)