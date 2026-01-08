import os
import json
import argparse

from methods import methods_call

def add_layerprune_args(parser: argparse.ArgumentParser):
    """base arguments"""

    group = parser.add_argument_group('base', 'base configuration')
    group.add_argument('--method', type=str, choices=list(methods_call.keys()), help="The method name ['sleb', 'mka', 'shortgpt', 'reverse', 'taylor', 'magnitude', 'laco', 'concat_merge']")
    group.add_argument("--model-name", type=str, help="A huggingface model name or a hf model path")
    group.add_argument("--target-layers", type=int, help="The final number of layers in the model")
    group.add_argument("--save-path", type=str, default=None, help="The path to save pruning information")
    group.add_argument("--continue-saving", action='store_true', help="Save each model generated at every pruning step")
    group.add_argument("--ppl-data", nargs='+', choices=["c4", "wiki2"], default=[], help="Test in the end")
    group.add_argument("--seed", type=int, default=10, help="Random seed")
    return parser


def add_sleb_args(parser: argparse.ArgumentParser):
    '''
    SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks
    https://github.com/jiwonsong-dev/SLEB
    https://arxiv.org/abs/2402.09025
    '''
    group = parser.add_argument_group('sleb', 'sleb configurations')
    group.add_argument("--calibration-dataset", choices=["c4", "wiki2", "pg19", "bookcorpus", "alpaca", "mmlu"], default="wiki2", help="The calibration dataset")
    group.add_argument("--nsamples", type=int, default=128, help="The number of samples in the calibration dataset used")
    return parser


def add_shortgpt_args(parser: argparse.ArgumentParser):
    '''
    ShortGPT: Layers in Large Language Models are More Redundant Than You Expect
    https://github.com/sramshetty/ShortGPT
    https://arxiv.org/abs/2403.03853
    '''
    group = parser.add_argument_group('shortgpt', 'shortgpt configurations')
    group.add_argument("--calibration-dataset", choices=["c4", "wiki2", "pg19", "bookcorpus", "alpaca", "mmlu"], default="pg19", help="The calibration dataset")
    group.add_argument("--nsamples", type=int, default=256, help="The number of samples in the calibration dataset used")
    return parser

def add_magnitude_args(parser: argparse.ArgumentParser):
    '''
    Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods
    https://github.com/Nota-NetsPresso/shortened-llm
    '''
    group = parser.add_argument_group('magnitude', 'magnitude configurations')
    group.add_argument("--calibration-dataset", choices=["c4", "wiki2", "pg19", "bookcorpus", "alpaca", "mmlu"], default="wiki2", help="The calibration dataset")
    group.add_argument("--nsamples", type=int, default=128, help="The number of samples in the calibration dataset used")
    group.add_argument("--weight-reduction", type=str, choices=["sum", "mean", "max", "prob"], default="sum", help="weight reduction")
    group.add_argument("--block-reduction", type=str, choices=["sum", "mean", "max", "prob"], default="sum", help="block reduction")
    group.add_argument("--heuristic", action='store_true', help="magnitude+")
    return parser

def add_taylor_args(parser: argparse.ArgumentParser):
    '''
    Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods
    https://github.com/Nota-NetsPresso/shortened-llm
    '''
    group = parser.add_argument_group('taylor', 'taylor configurations')
    group.add_argument("--calibration-dataset", choices=["c4", "wiki2", "pg19", "bookcorpus", "alpaca", "mmlu"], default="wiki2", help="The calibration dataset")
    group.add_argument("--nsamples", type=int, default=128, help="The number of samples in the calibration dataset used")
    group.add_argument("--weight-reduction", type=str, choices=["sum", "mean", "max", "prob"], default="sum", help="weight reduction")
    group.add_argument("--block-reduction", type=str, choices=["sum", "mean", "max", "prob"], default="sum", help="block reduction")
    group.add_argument("--heuristic", action='store_true', help="taylor+")
    return parser

def add_mka_args(parser: argparse.ArgumentParser):
    '''
    Pruning via Merging: Compressing LLMs via Manifold Alignment Based Layer Merging
    https://github.com/SempraETY/Pruning-via-Merging
    https://aclanthology.org/2024.emnlp-main.987/
    '''
    group = parser.add_argument_group('mka', 'mka configurations')
    group.add_argument("--calibration-dataset", choices=["c4", "wiki2", "pg19", "bookcorpus", "alpaca", "mmlu"], default="mmlu", help="The calibration dataset")
    group.add_argument("--nsamples", type=int, default=250, help="The number of samples in the calibration dataset used.")
    group.add_argument("--num-tasks", type=int, default=50, help="How many categories are divided.")
    return parser


def add_reverse_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("reverse", "reverse configurations")
    group.add_argument('--retain-layer', type=int, nargs='+', default=[], help='The pruning order of the retained layers is by default placed before layer 0')
    return parser

def add_concat_merge_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('concat_merge', 'concat_merge configurations')
    group.add_argument("--calibration-dataset", choices=["c4", "wiki2", "pg19", "bookcorpus", "alpaca", "mmlu"], default="wiki2", help="The calibration dataset")
    group.add_argument("--skip-method", choices=["bi", "mka", "sleb"], default="bi", help="The calibration dataset")
    group.add_argument("--nsamples", type=int, default=256, help="The number of samples in the calibration dataset used")
    group.add_argument("--merge-item", type=int, default=2, help="The number of blocks integrated for each item")
    group.add_argument("--wo-repeat", action='store_true', help="The iterative calculation of parameter importance and skip importance.")
    group.add_argument("--psu", type=int, default=1)
    group.add_argument("--min-max", type=float, default=0.0)


    return parser

def add_concat_merge_P_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('concat_merge_P', 'Posterior-based concat_merge configurations')
    group.add_argument("--calibration-dataset", choices=["c4", "wiki2", "pg19", "bookcorpus", "alpaca", "mmlu"], default="wiki2", help="The calibration dataset")
    group.add_argument("--skip-method", choices=["bi", "mka", "sleb"], default="bi", help="The calibration dataset")
    group.add_argument("--nsamples", type=int, default=256, help="The number of samples in the calibration dataset used")
    group.add_argument("--merge-item", type=int, default=2, help="The number of blocks integrated for each item")
    group.add_argument("--wo-repeat", action='store_true', help="The iterative calculation of parameter importance and skip importance.")
    group.add_argument("--granularity", type=int, default=20, help="Search granularity")
    return parser

ADD_METHODS_ARGS = {"sleb": add_sleb_args,
                    "mka": add_mka_args,
                    "shortgpt": add_shortgpt_args,
                    "reverse": add_reverse_args,
                    "magnitude": add_magnitude_args,
                    "taylor": add_taylor_args,
                    "concat_merge": add_concat_merge_args,
                    "concat_merge_P": add_concat_merge_P_args,
                    }

def get_args():
    parser = argparse.ArgumentParser(description='Layer Pruning')
    parser = add_layerprune_args(parser)
    args, unknown = parser.parse_known_args()
    if args.method in ADD_METHODS_ARGS:
        parser = ADD_METHODS_ARGS[args.method](parser)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    # parser = globals()[f"add_{args.method}_args"](parser)
    args, unknown = parser.parse_known_args()

    if hasattr(args, "heuristic"):
        args.method = args.method + "+"

    assert all(["--" not in x for x in unknown]), unknown

    args.model_name = args.model_name.rstrip('/\\')

    args.save_name = "_".join([os.path.basename(args.model_name), args.method])

    os.makedirs(args.save_path, exist_ok=True)
    args_dict = vars(args)

    with open(os.path.join(args.save_path, f'{args.method}_args.json'), 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

    return args
