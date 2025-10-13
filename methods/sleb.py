import os
import time
import json
import copy
from tqdm import tqdm

import torch
import torch.nn as nn

from utils.data_utils import get_trainloaders
from utils.eval_utils import load_and_eval_ppl, eval_zero_shot

@torch.inference_mode()
@torch.no_grad()
def get_loss(model, dataloader, bs=1, device=None):
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


def main_func(args, modelhander):
    modelhander.model.eval()

    dataloader = get_trainloaders(args.calibration_dataset,
                                  tokenizer=modelhander.tokenizer,
                                  nsamples=args.nsamples,
                                  seed=args.seed,
                                  seqlen=modelhander.model.seqlen
                                  )
    logging.info(f"Dataloader({args.calibration_dataset}) loaded.")

    num_remove_blocks = modelhander.config.num_hidden_layers - args.target_layers
    removal_list = []
    sleb_info = {}
    alive_list = list(range(modelhander.config.num_hidden_layers))
    none_dict = {i: None for i in alive_list}
    # check start time
    start_point = time.time()
    for i in range(num_remove_blocks):
        loss_dict = copy.copy(none_dict)
        phase_start_point = time.time()
        logging.info(f"Phase {i+1} of {num_remove_blocks}")

        min_loss = float("inf")
        min_loss_idx = -1

        for j in range(modelhander.model.config.num_hidden_layers):

            # kill j-th alive block
            del_layers = modelhander.remove_layers(j, ruturn_dict=True)

            loss = get_loss(modelhander.model, dataloader, bs=1, device=torch.device("cuda:0"))
            torch.cuda.empty_cache()

            loss_dict[alive_list[j]] = loss
            
            if loss < min_loss:
                min_loss = loss
                min_loss_idx = j

            logging.info(
                f"[Block {j} (Original block {alive_list[j]}) removed] Loss={loss:.3f}, Current Min Loss={min_loss:.3f} / Layer {alive_list[min_loss_idx]}"
            )

            # unkill j-th alive block
            modelhander.add_layers(del_layers)

        phase_time_elapsed = time.time() -  phase_start_point

        # remove block causing the least snlls increase
        logging.info(f"Phase_time_elapsed (s): {phase_time_elapsed}")
        logging.info(f"[SELECTED block {min_loss_idx} (Originally block {alive_list[min_loss_idx]})] Loss={min_loss:.3f}")

        modelhander.remove_layers(min_loss_idx)
        removal_list.append(alive_list[min_loss_idx])
        logging.info(f"Current Block Removal List: {removal_list}")
        del alive_list[min_loss_idx]
        loss_dict["remove_order"] = copy.copy(removal_list)
        loss_dict["time_using"] = phase_time_elapsed
        sleb_info[f"Phase_{i+1}"] = loss_dict

        if args.continue_saving:
            save_path = os.path.join(args.save_path, args.save_name + f"_{modelhander.model.config.num_hidden_layers}")
            modelhander.save(path=save_path)
            logging.info(f"Save model to {save_path}")

            for da in args.ppl_data:
                logging.info(f"Starting {da} PPL evaluation...")
                ppl = load_and_eval_ppl(modelhander.model, dataset=da, tokenizer=modelhander.tokenizer)
                loss_dict[f"{da}_ppl"] = ppl

    time_elapsed = time.time() - start_point
    if not args.continue_saving:
        save_path = os.path.join(args.save_path, args.save_name + f"_{modelhander.model.config.num_hidden_layers}")
        modelhander.save(path=save_path)

        for da in args.ppl_data:
            logging.info(f"Starting {da} PPL evaluation...")
            ppl = load_and_eval_ppl(modelhander.model, dataset=da, tokenizer=modelhander.tokenizer)
            sleb_info[f"{da}_ppl"] = ppl

    logging.info(f"Save {args.method} information to {os.path.join(args.save_path, f'{args.method}_info.json')}")
    with open(os.path.join(args.save_path, f'{args.method}_info.json'), 'w') as json_file:
        json.dump(sleb_info, json_file, indent=4)

    logging.info(
        f"Time_Elapsed: {time_elapsed}\n"
        f"Model Name: {args.model_name}\n"
        f"# Total Blocks: {modelhander.config.num_hidden_layers}\n"
        f"# Remove Blocks: {num_remove_blocks}\n"
        f"Calibration Dataset: {args.calibration_dataset}\n"
        f"Seed: {args.seed}\n" 
        f"Block Removal Order: {removal_list}\n"
    )


