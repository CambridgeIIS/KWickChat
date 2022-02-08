from train import *
from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                  GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
logger = logging.getLogger(__file__)

parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="",
                    help="Path or url of the dataset. If empty download from S3.")
parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
parser.add_argument("--model_checkpoint", type=str, default="openai-gpt", help="Path, url or short name of the model")
parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument("--personality_permutations", type=int, default=1,
                    help="Number of permutations of personality sentences")
parser.add_argument("--eval_before_start", action='store_true',
                    help="If true start with a first evaluation before training")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device (cuda or cpu)")
parser.add_argument("--fp16", type=str, default="",
                    help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
parser.add_argument("--local_rank", type=int, default=-1,
                    help="Local rank for distributed training (-1: not distributed)")
parser.add_argument("--distributed", type=int, default=0,
                    help="")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)



tokenizer_class = GPT2Tokenizer if "gpt2" in args.model_checkpoint else OpenAIGPTTokenizer
print('get tokenizer')
tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
print('prepare dataloader')
train_loader, valid_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)

valid_data = next(iter(valid_loader))

print(np.save('valid_data',valid_data))


print(valid_loader)
# print()