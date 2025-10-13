from utils.model_utils import get_llmhander
from utils.argments_utils import get_args
from utils.util import set_seed, init_logging
from methods import methods_call

init_logging()
args = get_args()
set_seed(args.seed)
modelhander= get_llmhander(args.model_name, concat_merge=(args.method=="concat_merge"))
methods_call[args.method](args, modelhander)

