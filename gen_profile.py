
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import pipeline, BitsAndBytesConfig
import argparse
from rank_bm25 import BM25Okapi
# from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import transformers
import json
from utils import split_batch, get_first_k_tokens, print_trainable_parameters, name2taskid
from utils import extract_citation_title, extract_option, extract_movie, extract_news_cat, extract_news_headline, extract_product_review, extract_scholarly_title, extract_tweet_paraphrasing

parser = argparse.ArgumentParser(description="Parser for LoRA")
parser.add_argument('--model_name', type=str, default='lmsys/vicuna-7b-v1.5')
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--cut_off', type=int, default=2048)
parser.add_argument('--task_name', type=str, default='movie_tagging')

args = parser.parse_args()
model_name = args.model_name
batch_size = args.batch_size
k = args.k
cutoff_len = args.cut_off
add_eos_token = False


# # 4 bit quantization inference  
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
#     max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
# )

# 8-bit quantization inference
# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     bnb_8bit_quant_type="nf8",
#     bnb_8bit_compute_dtype=torch.float16,
#     bnb_8bit_use_double_quant=True,
#    max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
# )

# 16-bit quantization inference
# bnb_config = BitsAndBytesConfig(
#     load_in_16bit=True,
#     bnb_16bit_quant_type="bf16",
#     bnb_16bit_compute_dtype=torch.bfloat16,
#     bnb_16bit_use_double_quant=True,
#     max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
# )

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.eos_token = "</s>"
tokenizer.pad_token = '[PAD]'

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    local_files_only=False,
    device_map='auto',
    trust_remote_code=True,
    torch_dtype=torch.float16
)

base_model.config.use_cache = True
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.eos_token_id = tokenizer.eos_token_id
base_model.config.bos_token_id = tokenizer.bos_token_id


with open(f"./data/{args.task_name}/user_others.json", 'r') as f:
    train_data = json.load(f)

with open(f"./data/{args.task_name}/user_top_100_history.json", 'r') as f:
    test_data = json.load(f)

with open('./prompt/prompt_profile.json', 'r') as f:
    prompt_template = json.load(f)

from tqdm import tqdm
import random

K = args.k

prompt_list_others = []
userid_list_others = []

for user in tqdm(train_data):

    history_list = []
    
    if len(user['profile'])> K:
        profiles = random.sample(user['profile'], K)
    else:
        profiles = user['profile']

    
    for p in profiles:
        for key, value in p.items():
            p[key] = get_first_k_tokens(p[key], 200)

    for p in profiles:
        history_list.append(prompt_template[args.task_name]['retrieval_history'].format(**p))

    history_string = ' | '.join(history_list)

    test_prompt = prompt_template[args.task_name]["profile_prompt"].format(history_list)

    prompt_list_others.append(test_prompt)
    userid_list_others.append(user['user_id'])


prompt_list_100 = []
userid_list_100 = []

for user in tqdm(test_data):

    history_list = []

    if len(user['profile'])> K:
        profiles = random.sample(user['profile'], K)
    else:
        profiles = user['profile']

    for p in profiles:
        for key, value in p.items():
            p[key] = get_first_k_tokens(p[key], 200)

    for p in profiles:
        history_list.append(prompt_template[args.task_name]['retrieval_history'].format(**p))

    history_string = ' | '.join(history_list)

    test_prompt = prompt_template[args.task_name]['profile_prompt'].format(history_list)
    prompt_list_100.append(test_prompt)
    userid_list_100.append(user['user_id'])

batched_prompt_others = split_batch(prompt_list_others, batch_size)
out_list_others = []

print(len(prompt_list_others))
print(len(batched_prompt_others))

with torch.no_grad():
    for batch_idx, batch in tqdm(enumerate(batched_prompt_others), total=len(batched_prompt_others)):
        # try:
        sentences = batch
        inputs = tokenizer(sentences, return_tensors="pt", padding=True, return_token_type_ids=False)
        inputs = inputs.to(base_model.device)

        with torch.autocast(device_type="cuda"):
            outputs = base_model.generate(
                **inputs,
                do_sample=True,
                top_k=10,
                temperature=0.6,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=300,
            )

        out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        out_list_others += out_sentence
        # except:
        #     out_list_others += ['']

pred_all_others = []

for i in range(len(out_list_others)):
    output = out_list_others[i].replace(prompt_list_others[i], '')
    pred_all_others.append({
        "id": userid_list_others[i],
        "output": output
        })
    
    print(output)


with open(f'./data/{args.task_name}/profile_user_others.json', 'w') as f:
    json.dump(pred_all_others, f)


batched_prompt_100 = split_batch(prompt_list_100, batch_size)
out_list_100 = []

with torch.no_grad():
    for batch_idx, batch in tqdm(enumerate(batched_prompt_100), total=len(batched_prompt_100)):
        sentences = batch
        inputs = tokenizer(sentences, return_tensors="pt", padding=True, return_token_type_ids=False)
        inputs = inputs.to(base_model.device)

        with torch.autocast(device_type="cuda"):
            outputs = base_model.generate(
                **inputs,
                do_sample=True,
                top_k=10,
                temperature=0.6,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=200,
            )

        out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        out_list_100 += out_sentence
        # except:
        #     out_list_100 += ['']

pred_all_100 = []

for i in range(len(out_list_100)):
    output = out_list_100[i].replace(prompt_list_100[i], '')
    pred_all_100.append({
        "id": userid_list_100[i],
        "output": output
        })
    
    print(output)


with open(f'./data/{args.task_name}/profile_user_100.json', 'w') as f:
    json.dump(pred_all_100, f)
