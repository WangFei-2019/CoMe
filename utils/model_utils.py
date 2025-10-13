from typing import List, Optional

import torch
from torch.nn import DataParallel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.utils.data import DataLoader

from collections import OrderedDict
import numpy as np
from tqdm import tqdm

from .util import * 

def get_llmhander(model_name, device_map="auto", layers_name=None, concat_merge=False):
    logging.info(f"Loaded Model: {model_name}")

    modelhander = ModelHandler(model_name=model_name, device_map=device_map, layers_name=layers_name, merge_method=concat_merge)
    modelhander.model.seqlen = 2048 # 2048
    modelhander.model.name = model_name

    return modelhander

    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 torch_dtype='auto',
                                                 low_cpu_mem_usage=True,
                                                 device_map=device_map,
                                                 )
    model.seqlen = 2048
    model.name = model_name

    


class ModelHandler():
    def __init__(self, 
                 model_name: str, 
                 layers_name: str = "model.layers",
                 device_map: str = "auto",
                 merge_method: bool = False
                 ):
        """
        HuggingFace Model Wrapper

        Args:
            model_name (str): HuggingFace model name
            layers_path (str): String in dot notation demonstrating how to access layers of the model. Ex: "model.layers"
            (Optional) merge_item (int): N to 1.
        """
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        if "llama" in config.architectures[0].lower() or "vicuna" in config.architectures[0].lower():
            if merge_method:
                from models_unit.llama.modeling_llama import LlamaForCausalLM as ModelForCausalLM
            else:
                from transformers import LlamaForCausalLM as ModelForCausalLM
            layers_name = "model.layers"
            self.merge_heads = self.merge_heads_llama
            self.merge_neuron = self.merge_neuron_llama
            self.head_name = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight"]
            self.ffn_name = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight"]
        elif "opt" in config.architectures[0].lower():
            if merge_method:
                from models_unit.opt.modeling_opt import OPTForCausalLM as ModelForCausalLM
            else:
                from transformers import OPTForCausalLM as ModelForCausalLM
            layers_name = "model.decoder.layers"
            self.merge_heads = self.merge_heads_opt
            self.merge_neuron = self.merge_neuron_opt
            self.head_name = ["self_attn.q_proj.weight", "self_attn.q_proj.bias", "self_attn.k_proj.weight", "self_attn.k_proj.bias", "self_attn.v_proj.weight", "self_attn.v_proj.bias", "self_attn.out_proj.weight", "self_attn.out_proj.bias"]
            self.ffn_name = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "self_attn_layer_norm.weight", "self_attn_layer_norm.bias", "final_layer_norm.weight", "final_layer_norm.bias"]
        elif "qwen2" in config.architectures[0].lower():
            if merge_method:
                from models_unit.qwen2.modeling_qwen2 import Qwen2ForCausalLM as ModelForCausalLM
            else:
                from transformers import Qwen2ForCausalLM as ModelForCausalLM
            layers_name = "model.layers"
            self.merge_heads = self.merge_heads_llama
            self.merge_neuron = self.merge_neuron_llama
            self.head_name = ["self_attn.q_proj.weight", "self_attn.q_proj.bias", "self_attn.k_proj.weight", "self_attn.k_proj.bias", "self_attn.v_proj.weight", "self_attn.v_proj.bias", "self_attn.o_proj.weight"]
            self.ffn_name = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight"]

        elif "qwen3" in config.architectures[0].lower():
            if merge_method:
                from models_unit.qwen3.modeling_qwen3 import Qwen3ForCausalLM as ModelForCausalLM
            else:
                from transformers import Qwen3ForCausalLM as ModelForCausalLM
            layers_name = "model.layers"
            self.merge_heads = self.merge_heads_qwen3
            self.merge_neuron = self.merge_neuron_llama
            self.head_name = ["self_attn.q_proj.weight", "self_attn.q_proj.bias", "self_attn.k_proj.weight", "self_attn.k_proj.bias", "self_attn.v_proj.weight", "self_attn.v_proj.bias", "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight"]
            self.ffn_name = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight"]

        elif "mistral" in config.architectures[0].lower():
            if merge_method:
                from models_unit.mistral.modeling_mistral import MistralForCausalLM as ModelForCausalLM
            else:
                from transformers import MistralForCausalLM as ModelForCausalLM
            layers_name = "model.layers"
            self.merge_heads = self.merge_heads_llama
            self.merge_neuron = self.merge_neuron_llama
            self.head_name = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight"]
            self.ffn_name = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight"]
        else:
            from transformers import AutoModelForCausalLM as ModelForCausalLM

        self.model = ModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map=device_map, trust_remote_code=True, config=config, )

        if not hasattr(self.model.config, 'head_dim') or self.model.config.head_dim is None:
            self.model.config.head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) # AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = self.model.config

        modules = layers_name.split(".")
        mod = self.model
        for m in modules:
            mod = getattr(mod, m)
        self.layers = mod
        self.base_layer_len = len(self.layers)

    @torch.no_grad
    def add_heads(self, merge_list: list, ratio: list, *args, **kwargs):
        state_dicts = []
        for layer_idx, r in zip(merge_list, ratio):
            # Create an OrderedDict for each layer
            layer_state_dict = OrderedDict()
            # Store the processed parameters in the OrderedDict
            for name in self.head_name:
                # layer_state_dict[name] = get_nested_attribute(self.layers[layer_idx], name) * r
                # -----------------------------------------
                # For MKA
                if "bias" in name:
                    if name not in layer_state_dict:
                        layer_state_dict[name] = get_nested_attribute(self.layers[layer_idx], name)
                else:
                    layer_state_dict[name] = get_nested_attribute(self.layers[layer_idx], name) * r
                # ------------------------------------------------

            state_dicts.append(layer_state_dict)
    
        merged_state_dict = OrderedDict()
        device = get_nested_attribute(self.layers[merge_list[0]], self.head_name[0]).device

        for key in state_dicts[0].keys():
            params = [sd[key] for sd in state_dicts]
            if any(param is not None for param in params):
                merged_state_dict[key] = torch.nn.Parameter(
                    torch.sum(
                        torch.stack([param.to(device) for param in params if param is not None]),
                        dim=0
                    )
                )
        return merged_state_dict
    
    @torch.no_grad
    def add_neuron(self, merge_list: list, ratio: list, *args, **kwargs):
        state_dicts = []
        for layer_idx, r in zip(merge_list, ratio):
            layer_state_dict = OrderedDict()
            for name in self.ffn_name:
                layer_state_dict[name] = get_nested_attribute(self.layers[layer_idx], name) * r
                # # -----------------------------------------
                # # For MKA
                # if "norm" in name:
                #     if name not in layer_state_dict:
                #         layer_state_dict[name] = get_nested_attribute(self.layers[layer_idx], name)
                # else:
                #     layer_state_dict[name] = get_nested_attribute(self.layers[layer_idx], name) * r
                # # ------------------------------------------------
            state_dicts.append(layer_state_dict)

        merged_state_dict = OrderedDict()
        device = get_nested_attribute(self.layers[merge_list[0]], self.ffn_name[0]).device

        for key in state_dicts[0].keys():
            params = [sd[key] for sd in state_dicts]
            if any(param is not None for param in params):
                merged_state_dict[key] = torch.nn.Parameter(
                    torch.sum(
                        torch.stack([param.to(device) for param in params if param is not None]),
                        dim=0
                    )
                )
        return merged_state_dict

    def laco_merge():
        pass

    @torch.no_grad
    def merge_heads_llama(self, merge_list: list, ratio: list, headgroup_importance):
        headgroup_to_remove = [np.argsort(np.array(i)).tolist() for i in headgroup_importance]
        num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads
        merge_num = distribute_and_round(self.config.num_key_value_heads, ratio)

        state_dicts = []
        for layer_idx, num in zip(merge_list, merge_num):
            logging.info(f"Layer {layer_idx} heads reserve: {num}")
            if num==0:
                continue
            head_size = self.layers[layer_idx].self_attn.k_proj.weight.size(0)
            head_dim = self.config.head_dim if hasattr(self.config, 'head_dim') else head_size // self.config.num_attention_heads
            mask = torch.zeros(head_size, dtype=torch.bool)

            for head_index in headgroup_to_remove[layer_idx][-num:]:
                start = head_index * head_dim
                end = start + head_dim
                mask[start:end] = 1

            # Create an OrderedDict for each layer
            layer_state_dict = OrderedDict()

            # Store the processed parameters in the OrderedDict
            layer_state_dict['self_attn.q_proj.weight'] = self.layers[layer_idx].self_attn.q_proj.weight[mask.repeat(num_key_value_groups), :]
            layer_state_dict['self_attn.q_proj.bias'] = self.layers[layer_idx].self_attn.q_proj.bias[mask.repeat(num_key_value_groups)] if self.layers[layer_idx].self_attn.q_proj.bias is not None else None

            layer_state_dict['self_attn.o_proj.weight'] = self.layers[layer_idx].self_attn.o_proj.weight[:, mask.repeat(num_key_value_groups)]

            layer_state_dict['self_attn.k_proj.weight'] = self.layers[layer_idx].self_attn.k_proj.weight[mask, :]
            layer_state_dict['self_attn.k_proj.bias'] = self.layers[layer_idx].self_attn.k_proj.bias[mask] if self.layers[layer_idx].self_attn.k_proj.bias is not None else None

            layer_state_dict['self_attn.v_proj.weight'] = self.layers[layer_idx].self_attn.v_proj.weight[mask, :]
            layer_state_dict['self_attn.v_proj.bias'] = self.layers[layer_idx].self_attn.v_proj.bias[mask] if self.layers[layer_idx].self_attn.v_proj.bias is not None else None
            state_dicts.append(layer_state_dict)

        # Merge weights across layers
        merged_state_dict = OrderedDict()
        device = self.layers[merge_list[0]].self_attn.q_proj.weight.device

        for key in state_dicts[0].keys():
            params = [sd[key] for sd in state_dicts]
            if any(param is not None for param in params):
                merged_state_dict[key] = torch.nn.Parameter(
                    torch.concat(
                        [param.to(device) for param in params if param is not None],
                        dim=-1 if "o_proj" in key else 0
                    )
                )
        if self.layers[layer_idx].self_attn.o_proj.bias is not None:
            o_bias = [self.layers[i].self_attn.o_proj.bias.to(device)*j for i,j in zip(merge_list, ratio)]
            merged_state_dict["self_attn.o_proj.bias"] = torch.nn.Parameter(torch.sum(torch.stack(o_bias), dim=0))

        logging.info("Finish Merging Head Groups")
        return merged_state_dict

    @torch.no_grad
    def merge_neuron_llama(self, merge_list: list, ratio: list, neuron_importance):
        neuron_to_remove = [np.argsort(np.array(i)).tolist() for i in neuron_importance]
        intermediate_size = self.config.intermediate_size
        merge_num = distribute_and_round(intermediate_size, ratio)

        state_dicts = []

        for layer_idx, num in zip(merge_list, merge_num):
            # random.shuffle(neuron_to_remove[layer_idx])
            reserve = neuron_to_remove[layer_idx][-num:]
            logging.info(f"Layer {layer_idx} neuron reserve: {num}")
            if num==0:
                continue
            mask = torch.zeros(intermediate_size, dtype=torch.bool)
            mask[reserve] = 1

            # Create an OrderedDict for each layer
            layer_state_dict = OrderedDict()

            # Store the processed parameters in the OrderedDict
            layer_state_dict['mlp.gate_proj.weight'] = self.layers[layer_idx].mlp.gate_proj.weight[mask, :]
            layer_state_dict['mlp.up_proj.weight'] = self.layers[layer_idx].mlp.up_proj.weight[mask, :]
            layer_state_dict['mlp.down_proj.weight'] = self.layers[layer_idx].mlp.down_proj.weight[:, mask]

            state_dicts.append(layer_state_dict)

        # Merge weights across layers
        merged_state_dict = OrderedDict()
        device = self.layers[merge_list[0]].mlp.gate_proj.weight.device

        for key in state_dicts[0].keys():
            params = [sd[key] for sd in state_dicts]
            if any(param is not None for param in params):
                merged_state_dict[key] = torch.nn.Parameter(
                    torch.concat(
                        [param.to(device) for param in params if param is not None],
                        dim=-1 if "down" in key else 0
                    )
                )

        if self.layers[layer_idx].mlp.down_proj.bias is not None:
            down_bias = [self.layers[i].mlp.down_proj.bias.to(device)*j for i,j in zip(merge_list, ratio)]
            merged_state_dict["mlp.down_proj.bias"] = torch.nn.Parameter(torch.sum(torch.stack(down_bias), dim=0))

        # Update layernorm weights
        ln_w = [self.layers[i].input_layernorm.weight.to(device) for i in merge_list]
        merged_state_dict["input_layernorm.weight"] = torch.nn.Parameter(torch.mean(torch.stack(ln_w), dim=0))

        # device = self.layers[merge_list[0]].post_attention_layernorm.weight.device
        ln_w = [self.layers[i].post_attention_layernorm.weight.to(device) for i in merge_list]
        merged_state_dict["post_attention_layernorm.weight"] = torch.nn.Parameter(torch.mean(torch.stack(ln_w), dim=0))

        # ln_w = [self.layers[i].input_layernorm.weight.to(device)*j for i, j in zip(merge_list, ratio)]
        # merged_state_dict["input_layernorm.weight"] = torch.nn.Parameter(torch.sum(torch.stack(ln_w), dim=0))

        # ln_w = [self.layers[i].post_attention_layernorm.weight.to(device)*j for i, j in zip(merge_list, ratio)]
        # merged_state_dict["post_attention_layernorm.weight"] = torch.nn.Parameter(torch.sum(torch.stack(ln_w), dim=0))

        logging.info("Finish Merging FFN Intermediate Dimension")

        return merged_state_dict

    @torch.no_grad
    def merge_heads_qwen3(self, merge_list: list, ratio: list, headgroup_importance):
        headgroup_to_remove = [np.argsort(np.array(i)).tolist() for i in headgroup_importance]
        num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads
        merge_num = distribute_and_round(self.config.num_key_value_heads, ratio)

        state_dicts = []
        for layer_idx, num in zip(merge_list, merge_num):
            logging.info(f"Layer {layer_idx} heads reserve: {num}")
            if num==0:
                continue
            head_size = self.layers[layer_idx].self_attn.k_proj.weight.size(0)
            head_dim = self.config.head_dim if hasattr(self.config, 'head_dim') else head_size // self.config.num_attention_heads
            mask = torch.zeros(head_size, dtype=torch.bool)

            for head_index in headgroup_to_remove[layer_idx][-num:]:
                start = head_index * head_dim
                end = start + head_dim
                mask[start:end] = 1

            # Create an OrderedDict for each layer
            layer_state_dict = OrderedDict()

            # Store the processed parameters in the OrderedDict
            layer_state_dict['self_attn.q_proj.weight'] = self.layers[layer_idx].self_attn.q_proj.weight[mask.repeat(num_key_value_groups), :]
            layer_state_dict['self_attn.q_proj.bias'] = self.layers[layer_idx].self_attn.q_proj.bias[mask.repeat(num_key_value_groups)] if self.layers[layer_idx].self_attn.q_proj.bias is not None else None

            layer_state_dict['self_attn.o_proj.weight'] = self.layers[layer_idx].self_attn.o_proj.weight[:, mask.repeat(num_key_value_groups)]

            layer_state_dict['self_attn.k_proj.weight'] = self.layers[layer_idx].self_attn.k_proj.weight[mask, :]
            layer_state_dict['self_attn.k_proj.bias'] = self.layers[layer_idx].self_attn.k_proj.bias[mask] if self.layers[layer_idx].self_attn.k_proj.bias is not None else None

            layer_state_dict['self_attn.v_proj.weight'] = self.layers[layer_idx].self_attn.v_proj.weight[mask, :]
            layer_state_dict['self_attn.v_proj.bias'] = self.layers[layer_idx].self_attn.v_proj.bias[mask] if self.layers[layer_idx].self_attn.v_proj.bias is not None else None
            state_dicts.append(layer_state_dict)

        # Merge weights across layers
        merged_state_dict = OrderedDict()
        device = self.layers[merge_list[0]].self_attn.q_proj.weight.device

        for key in state_dicts[0].keys():
            params = [sd[key] for sd in state_dicts]
            if any(param is not None for param in params):
                merged_state_dict[key] = torch.nn.Parameter(
                    torch.concat(
                        [param.to(device) for param in params if param is not None],
                        dim=-1 if "o_proj" in key else 0
                    )
                )

        if self.layers[layer_idx].self_attn.o_proj.bias is not None:
            o_bias = [self.layers[i].self_attn.o_proj.bias.to(device)*j for i,j in zip(merge_list, ratio)]
            merged_state_dict["self_attn.o_proj.bias"] = torch.nn.Parameter(torch.sum(torch.stack(o_bias), dim=0))

        q_norm = [self.layers[i].self_attn.q_norm.weight.to(device)*j for i,j in zip(merge_list, ratio)]
        merged_state_dict["self_attn.q_norm.weight"] = torch.nn.Parameter(torch.sum(torch.stack(q_norm), dim=0))

        k_norm = [self.layers[i].self_attn.k_norm.weight.to(device)*j for i,j in zip(merge_list, ratio)]
        merged_state_dict["self_attn.k_norm.weight"] = torch.nn.Parameter(torch.sum(torch.stack(k_norm), dim=0))

        logging.info("Finish Merging Head Groups")
        return merged_state_dict

    @torch.no_grad
    def merge_heads_opt(self, merge_list: list, ratio: list, headgroup_importance):
        headgroup_to_remove = [np.argsort(np.array(i)).tolist() for i in headgroup_importance]
        num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads if hasattr(self.config, "num_key_value_heads") else 1
        merge_num = distribute_and_round(self.config.num_key_value_heads if hasattr(self.config, "num_key_value_heads") else self.config.num_attention_heads, ratio)

        state_dicts = []
        for layer_idx, num in zip(merge_list, merge_num):
            logging.info(f"Layer {layer_idx} heads reserve: {num}")
            if num==0:
                continue
            head_size = self.layers[layer_idx].self_attn.k_proj.weight.size(0)
            head_dim = self.config.head_dim if hasattr(self.config, 'head_dim') else head_size // self.config.num_attention_heads
            mask = torch.zeros(head_size, dtype=torch.bool)

            for head_index in headgroup_to_remove[layer_idx][-num:]:
                start = head_index * head_dim
                end = start + head_dim
                mask[start:end] = 1

            # Create an OrderedDict for each layer
            layer_state_dict = OrderedDict()

            # Store the processed parameters in the OrderedDict
            layer_state_dict['self_attn.q_proj.weight'] = self.layers[layer_idx].self_attn.q_proj.weight[mask.repeat(num_key_value_groups), :]
            layer_state_dict['self_attn.q_proj.bias'] = self.layers[layer_idx].self_attn.q_proj.bias[mask.repeat(num_key_value_groups)] if self.layers[layer_idx].self_attn.q_proj.bias is not None else None

            layer_state_dict['self_attn.out_proj.weight'] = self.layers[layer_idx].self_attn.out_proj.weight[:, mask.repeat(num_key_value_groups)]

            layer_state_dict['self_attn.k_proj.weight'] = self.layers[layer_idx].self_attn.k_proj.weight[mask, :]
            layer_state_dict['self_attn.k_proj.bias'] = self.layers[layer_idx].self_attn.k_proj.bias[mask] if self.layers[layer_idx].self_attn.k_proj.bias is not None else None

            layer_state_dict['self_attn.v_proj.weight'] = self.layers[layer_idx].self_attn.v_proj.weight[mask, :]
            layer_state_dict['self_attn.v_proj.bias'] = self.layers[layer_idx].self_attn.v_proj.bias[mask] if self.layers[layer_idx].self_attn.v_proj.bias is not None else None
        
            state_dicts.append(layer_state_dict)

        # Merge weights across layers
        merged_state_dict = OrderedDict()
        device = self.layers[merge_list[0]].self_attn.q_proj.weight.device

        for key in state_dicts[0].keys():
            params = [sd[key] for sd in state_dicts]
            if any(param is not None for param in params):
                merged_state_dict[key] = torch.nn.Parameter(
                    torch.concat(
                        [param.to(device) for param in params if param is not None],
                        dim=-1 if "out_proj" in key else 0
                    )
                )

        if self.layers[layer_idx].self_attn.out_proj.bias is not None:
            out_bias = [self.layers[i].self_attn.out_proj.bias.to(device)*j for i,j in zip(merge_list, ratio)]
            merged_state_dict["self_attn.out_proj.bias"] = torch.nn.Parameter(torch.sum(torch.stack(out_bias), dim=0))

        logging.info("Finish Merging Head Groups")

        return merged_state_dict

    @torch.no_grad
    def merge_neuron_opt(self, merge_list: list, ratio: list, neuron_importance):
        neuron_to_remove = [np.argsort(np.array(i)).tolist() for i in neuron_importance]

        intermediate_size = self.config.ffn_dim
        merge_num = distribute_and_round(intermediate_size, ratio)

        state_dicts = []

        for layer_idx, num in zip(merge_list, merge_num):
            reserve = neuron_to_remove[layer_idx][-num:]
            logging.info(f"Layer {layer_idx} neuron reserve: {num}")
            if num==0:
                continue
            mask = torch.zeros(intermediate_size, dtype=torch.bool)
            mask[reserve] = 1

            # Create an OrderedDict for each layer
            layer_state_dict = OrderedDict()

            # Store the processed parameters in the OrderedDict
            layer_state_dict['fc1.weight'] = self.layers[layer_idx].fc1.weight[mask, :]
            layer_state_dict['fc1.bias'] = self.layers[layer_idx].fc1.bias[mask] if self.layers[layer_idx].fc1.bias is not None else None

            layer_state_dict['fc2.weight'] = self.layers[layer_idx].fc2.weight[:, mask]

            state_dicts.append(layer_state_dict)

        # Merge weights across layers
        merged_state_dict = OrderedDict()
        device = self.layers[merge_list[0]].fc2.weight.device

        for key in state_dicts[0].keys():
            params = [sd[key] for sd in state_dicts]
            if any(param is not None for param in params):
                merged_state_dict[key] = torch.nn.Parameter(
                    torch.concat(
                        [param.to(device) for param in params if param is not None],
                        dim=-1 if "fc2" in key else 0
                    )
                )
                
        fc2_bias = [self.layers[i].fc2.bias.to(device)*j for i,j in zip(merge_list, ratio)]
        merged_state_dict["fc2.bias"] = torch.nn.Parameter(torch.sum(torch.stack(fc2_bias), dim=0))

        ln_w = [self.layers[i].self_attn_layer_norm.weight.to(device) for i in merge_list]
        merged_state_dict["self_attn_layer_norm.weight"] = torch.nn.Parameter(torch.mean(torch.stack(ln_w), dim=0))

        ln_bias = [self.layers[i].self_attn_layer_norm.bias.to(device) for i in merge_list]
        merged_state_dict["self_attn_layer_norm.bias"] = torch.nn.Parameter(torch.mean(torch.stack(ln_bias), dim=0))

        ln_w = [self.layers[i].final_layer_norm.weight.to(device) for i in merge_list]
        merged_state_dict["final_layer_norm.weight"] = torch.nn.Parameter(torch.mean(torch.stack(ln_w), dim=0))

        ln_bias = [self.layers[i].final_layer_norm.bias.to(device) for i in merge_list]
        merged_state_dict["final_layer_norm.bias"] = torch.nn.Parameter(torch.mean(torch.stack(ln_bias), dim=0))

        logging.info("Finish Merging FFN Intermediate Dimension")

        return merged_state_dict
    
    @torch.no_grad
    def adjust_layer_index(self, merge_index_list, state_dict=None, save_index=None):
        assert (state_dict is not None and save_index is None) or (state_dict is None and save_index is not None)

        if save_index is None:
            merge_index_list = sorted(merge_index_list)
            prune_index_list = merge_index_list[1:]
            self.layers[merge_index_list[0]].load_state_dict(state_dict)

        else:
            assert save_index in merge_index_list
            prune_index_list = sorted(merge_index_list)
            prune_index_list.remove(save_index)

        
        self.remove_layers(removal_list=prune_index_list)

    @torch.no_grad
    def remove_layers(self, removal_list, ruturn_dict=False):
        if isinstance(removal_list, int):
            removal_list = [removal_list]
        removal_list = sorted(removal_list, reverse=True)

        del_layers = {}
        for layer_idx in removal_list:
            try:
                if ruturn_dict:
                    del_layers[layer_idx] = self.layers[layer_idx]
                del self.layers[layer_idx]
            except:
                IndexError(f"layer {layer_idx} does not exist, function may have already been called")

        for layer_idx, module in enumerate(self.layers):
            module.self_attn.layer_idx = layer_idx

        self.model.config.num_hidden_layers = len(self.layers)

        return del_layers
    
    @torch.no_grad
    def add_layers(self, add_dict):
        idxs = sorted(list(add_dict.keys()))
        for idx in idxs:
            self.layers.insert(idx, add_dict[idx])

        for layer_idx, module in enumerate(self.layers):
            module.self_attn.layer_idx = layer_idx
        
        self.model.config.num_hidden_layers = len(self.layers)

    def save(self, path):
        self.tokenizer.save_pretrained(path)

        # generation_config = self.model.generation_config
        # # 或者，如果你不希望使用采样生成
        # generation_config.do_sample = False
        # generation_config.temperature = None
        # generation_config.top_p = None

        # # 重新验证配置
        # generation_config.validate()

        self.model.save_pretrained(path)