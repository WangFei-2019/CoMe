import os
import time
import json
from tqdm import tqdm

import torch
from torch import nn 


from utils.data_utils import get_trainloaders
from utils.eval_utils import load_and_eval_ppl, eval_zero_shot
from utils.util import *



def block_ratio(score: list, p, min_max=0.95):
    assert isinstance(score, list)
    
    # Step 1: Normalize the scores
    normalized = [x / sum(score) for x in score]
    
    # Step 2: Exponential scaling
    exp_scaled = [x ** p for x in normalized]

    # Step 3: Re-normalize after exponential scaling
    normalized = [x / sum(exp_scaled) for x in exp_scaled]
    
    # Step 4: Ensure the maximum element is at least min_max
    max_index = normalized.index(max(normalized))
    
    if normalized[max_index] < min_max:
        # Calculate the amount to distribute
        excess = min_max - normalized[max_index]
        normalized[max_index] = min_max
        
        # Distribute the excess across other elements
        total_other = sum(normalized) - min_max
        if total_other > 0:
            for i in range(len(normalized)):
                if i != max_index:
                    normalized[i] = normalized[i] * (1 - min_max) / total_other
        else:
            # If all other values are zero, just distribute equally
            num_other = len(normalized) - 1
            for i in range(len(normalized)):
                if i != max_index:
                    normalized[i] = (1 - min_max) / num_other

    return normalized

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

def compute_head_importance(embed: List[torch.Tensor], method: List[str]):
    headgroup_importance = []
    for j in range(len(embed)):
        layer_emb = embed[j]
        batch, s_len, groups, _, _ = layer_emb.size()
        layer_emb = layer_emb.reshape(batch, s_len, groups, -1)
        layer_emb = METHODS[method[1]](layer_emb, axis=-1)

        layer_emb = METHODS[method[0]](layer_emb, axis=1) # Sequence avearge
        # layer_emb = METHODS[method[0]](layer_emb, axis=0)  # Batch average

        headgroup_importance.append(layer_emb.cpu().detach())

    return headgroup_importance


def compute_importance(embed: List[torch.Tensor], method: List[str]):
    '''Param:
            method: List[str] → batch, sequence, embedding'''
    importance = []
    method = method[::-1]
    for j in range(len(embed)):
        layer_emb = embed[j]
        i = 0
        for m in method: # embedding, sequence, batch
            layer_emb = METHODS[m](layer_emb, axis=-(i+1)) # Sequence avearge
            if m in ["none", "norm", "softmax"]:
                i+=1
        importance.append(layer_emb.cpu().detach())
    return importance

@torch.inference_mode()
def get_loss(model, dataloader, bs=2, device=None):
    if device == None:
        device = model.device
    # Get input IDs
    testenc = dataloader.input_ids
    testattn = dataloader.attention_mask

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen
  
    # List to store negative log likelihoods
    losses = []
    #logging.info(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0, nsamples, bs):

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        attn = testattn[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        attn = attn.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs, attention_mask=attn).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        loss = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        losses.append(loss)

    # Compute sum of negative log_likelihood
    loss_sum = torch.stack(losses).sum()

    return loss_sum.item()


@torch.inference_mode()
def eval_importance(model, dataloader, bs=1, device=None, merge_item=2):
    if device == None:
        device = model.device
    # Get input IDs
    testenc = dataloader.input_ids
    testattn = dataloader.attention_mask

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    if nsamples==0:
        nsamples=1
        model.seqlen=testenc.numel()

    layer_importance = [0 for _ in range(model.config.num_hidden_layers)]
    layer_importance_skip = [0 for _ in range(model.config.num_hidden_layers - merge_item + 1)]

    neuron_importance = None
    headgroup_importance = None

    # Loop through each batch
    for i in tqdm(range(0, nsamples, bs), desc="Parameter Importance Score calculating ..."):
        # Calculate end index
        j = min(i+bs, nsamples)
        # Prepare inputs and move to device
        inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        attn = testattn[:, (i * model.seqlen):(j * model.seqlen)].to(model.device)
        attn = attn.reshape(j-i, model.seqlen)

        # Forward pass through the model
        outputs = model(inputs, 
                        attention_mask=attn,
                        output_hidden_states=True,
                        output_mlp_embed=True,
                        output_heads_embed=True,
                        )

        layer_importance = list_item_sum(compute_bi(outputs.hidden_states, angular=False), layer_importance)
        layer_importance_skip = list_item_sum(compute_bi(outputs.hidden_states, angular=True, n_prune_layers=2), layer_importance_skip)

        importance = compute_importance(outputs.mlp_embed, ["mean", "none"])
        importance = [i.cpu() for i in importance]
        
        if neuron_importance is None:
            neuron_importance = importance
        else:
            neuron_importance = concat_list(neuron_importance, importance, axis=0)

        importance = compute_head_importance(outputs.heads_embed, method=["mean", "mean"])
        importance = [i.cpu() for i in importance]
        if headgroup_importance is None:
            headgroup_importance = importance
        else:
            headgroup_importance = concat_list(headgroup_importance, importance, axis=0)

    layer_importance_average = [i/len(layer_importance) for i in layer_importance]
    layer_importance_skip_average = [i/len(layer_importance_skip) for i in layer_importance_skip]
    neuron_importance = compute_importance(neuron_importance, ["mean", "none"])
    headgroup_importance = compute_importance(headgroup_importance, ["mean", "none"])

    return {
        "layer_importance": layer_importance_average,
        "layer_importance_skip": layer_importance_skip_average,
        "neuron_importance": neuron_importance,
        "headgroup_importance": headgroup_importance,
    }

def repeat_skip_importance_deal(info_lst, merge_item=2):
    if not isinstance(info_lst, list):
        info_lst = [info_lst]

    list_indices = [index for index, element in enumerate(info_lst) if isinstance(element, list)]
    new_list_indices = set()
    for index in list_indices:
        for i in range(index - (merge_item - 1), index + merge_item - 1):
            new_list_indices.add(i)

    new_list_indices = {i for i in new_list_indices if i >= 0}
    return list(new_list_indices)


def count_elements(nested_list):
    count = 0
    for element in nested_list:
        if isinstance(element, list):
            count += count_elements(element)
        else:
            count += 1
    return count


def main_func(args, modelhander):

    dataloader = get_trainloaders(args.calibration_dataset,
                                  tokenizer=modelhander.tokenizer,
                                  nsamples=args.nsamples,
                                  seed=args.seed,
                                  seqlen=modelhander.model.seqlen
                                  )

    args.save_name = args.save_name + f"_{args.skip_method}"
    
    concat_merge_info = {}
    layer_info = [list(range(modelhander.config.num_hidden_layers))]

    while modelhander.config.num_hidden_layers > args.target_layers:
        # calculate score
        score = eval_importance(model=modelhander.model, dataloader=dataloader, bs=2, merge_item=args.merge_item)
        layer_importance = score["layer_importance"]

        if args.skip_method == "bi":
            layer_importance_skip = score["layer_importance_skip"]
        elif args.skip_method == "sleb":
            layer_importance_skip = []
            for l_idx in tqdm(range(modelhander.config.num_hidden_layers - args.merge_item + 1), desc="Skip-Layer Score calculating ..."):
                layer_state_dict = modelhander.remove_layers([l_idx+i for i in range(args.merge_item)], ruturn_dict=True)
                layer_importance_skip.append(get_loss(modelhander.model, dataloader))
                modelhander.add_layers(layer_state_dict)
        elif args.skip_method == "mka":
            layer_importance_skip = [0 for _ in range(modelhander.model.config.num_hidden_layers - args.merge_item + 1)]
            layer_importance_skip[-1] = 1
        else:
            raise NameError(f"{args.skip_method} is not an available calculation method.")
        
        if args.wo_repeat:
            list_indices = repeat_skip_importance_deal(layer_info[-1], merge_item=args.merge_item)
            for indice in list_indices:
                layer_importance_skip[indice] = float("inf")

        # get layers to merge, and the merging ratio
        tar_lay_item = max(args.target_layers, modelhander.config.num_hidden_layers - args.merge_item+1)
        phase_name = f"{modelhander.config.num_hidden_layers}_to_{tar_lay_item}"
        concat_merge_info[phase_name] = {}

        logging.info("*"*20, f"Pruning {modelhander.config.num_hidden_layers} to {tar_lay_item}", "*"*20)
        logging.info(f"Layer importance: {', '.join([f'{k}→{x:.2f}' for k, x in enumerate(layer_importance)])}")
        logging.info(f"Skip Layer importance: {', '.join([f'{k}→{x:.2f}' for k, x in enumerate(layer_importance_skip)])}")

        idx = layer_importance_skip.index(min(layer_importance_skip)) if layer_importance_skip != [] else 0
        merge_list = [i+idx for i in range(modelhander.config.num_hidden_layers - tar_lay_item + 1)]
        min_max = args.min_max**(1/(count_elements([layer_info[-1][i] for i in merge_list]) - 1))
        # min_max = args.min_max
        ratio = block_ratio(score=[layer_importance[i] for i in merge_list], p=args.psu, min_max=min_max)

        layer_info = leaf_node_generate(layer_info, merge_list)
        concat_merge_info[phase_name]["merge_info"] = layer_info[-1]
        concat_merge_info[phase_name]["merge_list"] = merge_list
        concat_merge_info[phase_name]["merge_ratio"] = ratio
        concat_merge_info[phase_name]["layer_importance_skip"] = layer_importance_skip
        concat_merge_info[phase_name]["layer_importance"] = layer_importance


        logging.info(f"Merge information: {layer_info[-1]} (Merge list {merge_list} → Ratio [{', '.join([f'{x:.2f}' for x in ratio])}])")
        
        # merge parameters
        state_dict = modelhander.merge_heads(merge_list, ratio, headgroup_importance=score["headgroup_importance"])
        state_dict.update(modelhander.merge_neuron(merge_list, ratio, neuron_importance=score["neuron_importance"]))

        modelhander.adjust_layer_index(merge_index_list=merge_list, state_dict=state_dict)

        torch.save(score["headgroup_importance"], os.path.join(args.save_path, f"{phase_name}-headgroup_importance.pt"))
        torch.save(score["neuron_importance"], os.path.join(args.save_path, f"{phase_name}-neuron_importance.pt"))

        if args.continue_saving:
            save_path = os.path.join(args.save_path, args.save_name + f"_{modelhander.model.config.num_hidden_layers}{'_worepeat' if args.wo_repeat else ''}")
            modelhander.save(path=save_path)
            logging.info(f"Save model to {save_path}")
            for da in args.ppl_data:
                logging.info(f"Starting {da} PPL evaluation...")
                ppl = load_and_eval_ppl(modelhander.model, dataset=da, tokenizer=modelhander.tokenizer)
                concat_merge_info[phase_name][f"{da}_ppl"] = ppl

    if not args.continue_saving:
        save_path = os.path.join(args.save_path, args.save_name + f"_{modelhander.model.config.num_hidden_layers}{'_worepeat' if args.wo_repeat else ''}")
        modelhander.save(path=save_path)
        logging.info(f"Save model to {save_path}")
        for da in args.ppl_data:
            logging.info(f"Starting {da} PPL evaluation...")
            ppl = load_and_eval_ppl(modelhander.model, dataset=da, tokenizer=modelhander.tokenizer)
            concat_merge_info[f"{da}_ppl"] = ppl
            
    concat_merge_info["merge_info"] = layer_info
    logging.info("Merge Process:")
    for i, ids in enumerate(layer_info):
        logging.info(f"{i:3d}: {ids}")
    
    info_path = os.path.join(args.save_path, f'{args.save_name}_info.json')
    logging.info(f"Save {args.method} information to {info_path}")
    with open(info_path, 'w') as json_file:
        json.dump(concat_merge_info, json_file, indent=4)
