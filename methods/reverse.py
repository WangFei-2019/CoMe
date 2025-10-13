import os
import json
from tqdm import tqdm

from utils.eval_utils import load_and_eval_ppl, eval_zero_shot
from utils.util import *

def main_func(args, modelhander):
    reverse_info = {}
    sorted_indices = list(range(modelhander.config.num_hidden_layers - 1, -1, -1))
    for idx in args.retain_layer:
        try:
            sorted_indices.remove(idx)
        except:
            raise ValueError(f"The retain layer idx {idx} is not in the model.")
        sorted_indices.append(idx)

    reverse_info["sorted_indices"] = sorted_indices
    logging.info(f"Remove order of layer index: \n{sorted_indices}")

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
            reverse_info[f"Layers_{modelhander.model.config.num_hidden_layers}"] = {"remove_layer_idx_list": copy.copy(removal_list)}
            logging.info(f"Save model to {save_path}")
            for da in args.ppl_data:
                logging.info(f"Starting {da} PPL evaluation...")
                ppl = load_and_eval_ppl(modelhander.model, dataset=da, tokenizer=modelhander.tokenizer)
                reverse_info[f"Layers_{modelhander.model.config.num_hidden_layers}"][f"{da}_ppl"] = ppl

    else:
        removal_list=sorted_indices[:num_remove_blocks]
        logging.info(f"Remove layer index list is {removal_list}")
        modelhander.remove_layers(removal_list=removal_list)
        save_path = os.path.join(args.save_path, args.save_name + f"_{modelhander.model.config.num_hidden_layers}")
        modelhander.save(path=save_path)
        logging.info(f"Save model to {save_path}")
        reverse_info[f"Layers_{modelhander.model.config.num_hidden_layers}"] = {"remove_layer_idx_list": removal_list}
        for da in args.ppl_data:
            logging.info(f"Starting {da} PPL evaluation...")
            ppl = load_and_eval_ppl(modelhander.model, dataset=da, tokenizer=modelhander.tokenizer)
            reverse_info[f"Layers_{modelhander.model.config.num_hidden_layers}"][f"{da}_ppl"] = ppl

    save_path = os.path.join(args.save_path, f"{'_'.join([f'{args.method}_info'] + [str(i) for i in args.retain_layer])}.json")
    logging.info(f"Save {args.method} information to {save_path}")
    with open(save_path, 'w') as json_file:
        json.dump(reverse_info, json_file, indent=4)