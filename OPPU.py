import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
# from transformers import pipeline, BitsAndBytesConfig
import argparse
from rank_bm25 import BM25Okapi
# from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import transformers
from utils import split_batch, get_first_k_tokens, print_trainable_parameters, name2taskid
from utils import extract_citation_title, extract_option, extract_movie, extract_news_cat, extract_news_headline, extract_product_review, extract_scholarly_title, extract_tweet_paraphrasing
import json
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, PeftModel


parser = argparse.ArgumentParser(description="Parser for LoRA")
parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--k', type=int, default=0)
parser.add_argument('--max_step', type=int, default=5000)
parser.add_argument('--cut_off', type=int, default=2048)
parser.add_argument('--max_epoch', type=int, default=2)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--task_name', type=str, default='movie_tagging')
parser.add_argument('--add_profile', action='store_true')
parser.add_argument('--task_lora', type=str, default='./ckpt/movie_tagging/k1-movie_tagging-Llama-2-7b-hf-task_LoRA_ckpt')
parser.add_argument('--access_token', type=str, default=None)

args = parser.parse_args()
model_name = args.model_name
task_name = args.task_name
batch_size = args.batch_size
k = args.k
# max_step = args.max_step
cutoff_len = args.cut_off
add_eos_token = False
max_epoch = args.max_epoch

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

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=args.access_token)
tokenizer.eos_token = "</s>"
tokenizer.pad_token = '[PAD]'
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    local_files_only=False,
    device_map='auto',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

base_model.config.use_cache = False
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.eos_token_id = tokenizer.eos_token_id
base_model.config.bos_token_id = tokenizer.bos_token_id


from peft import prepare_model_for_kbit_training

base_model.gradient_checkpointing_enable()
base_model = prepare_model_for_kbit_training(base_model)



from peft import LoraConfig, get_peft_model 

peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"], # , "k_proj", "out_proj"
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_arguments = transformers.TrainingArguments(
    output_dir='outputs/',
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    optim='adamw_torch',
    num_train_epochs=max_epoch,
    save_steps=1e9,
    logging_steps=50,
    learning_rate=1e-4,
    weight_decay=1e-2,
    bf16=True,
    max_grad_norm=0.3,
    # max_steps=max_step,
    warmup_ratio=0.1,
    group_by_length=True,
    lr_scheduler_type='linear',
    report_to='none',
)


with open(f"./data/{task_name}/user_top_100_history.json", 'r') as f:
    test_data = json.load(f)

format_flag = False
if args.task_name == "movie_tagging":
    extract_article = extract_movie
    format_flag = True
elif args.task_name == "news_categorize":
    extract_article = extract_news_cat
    format_flag = True
elif args.task_name == "news_headline":
    extract_article = extract_news_headline
    format_flag = True
elif args.task_name == "product_rating":
    extract_article = extrat_product_review
    format_flag = True
elif args.task_name == "scholarly_title":
    extract_article = extract_scholarly_title
    format_flag = True
elif args.task_name == "tweet_paraphrase":
    extract_article = extrat_tweet_paraphrasing


with open('./prompt/prompt.json', 'r') as f:
    prompt_template = json.load(f)


if args.add_profile:
    with open(f'./data/{task_name}/profile_user_100.json', 'r') as f:
        test_profile = json.load(f)


def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = data_point['full_prompt']
    tokenized_full_prompt = tokenize(full_prompt)
    # if not train_on_inputs:
    user_prompt = data_point['prompt']
    
    tokenized_user_prompt = tokenize(
        user_prompt, add_eos_token=add_eos_token
    )
    user_prompt_len = len(tokenized_user_prompt["input_ids"])

    if add_eos_token:
        user_prompt_len -= 1

    tokenized_full_prompt["labels"] = [
        -100
    ] * user_prompt_len + tokenized_full_prompt["labels"][
        user_prompt_len:
    ]  # could be sped up, probably
    return tokenized_full_prompt



# training
from datasets import load_dataset, Dataset
model = PeftModel.from_pretrained(model=base_model, model_id=args.task_lora, is_trainable=False)
base_model = model.merge_and_unload()
print_trainable_parameters(model)


pred_all = []
actual = []
train_data = []

for i in tqdm(range(len(test_data))):
    model = get_peft_model(base_model, peft_config)
    print_trainable_parameters(model)

    if args.add_profile:
        profile = test_profile[i]['output']

    for idx, q in enumerate(test_data[i]['profile']):
        for key, value in q.items():
            q[key] = get_first_k_tokens(q[key], 768)
            
        prompt = prompt_template[args.task_name]['OPPU_input'].format(**q)
        full_prompt = prompt_template[args.task_name]['OPPU_full'].format(**q)

        if k > 0 and idx != 0 and format_flag==True:
            visible_history_list = test_data[i]['profile'][:idx]

            for p in visible_history_list:
                for key, value in p.items():
                    p[key] = get_first_k_tokens(p[key], 768)

            history_list = [prompt_template[args.task_name]['retrieval_history'].format(**p) for p in visible_history_list]
            tokenized_corpus = [doc.split(" ") for doc in history_list]
            bm25 = BM25Okapi(tokenized_corpus)

            tokenized_query = prompt_template[args.task_name]["retrieval_query"].format(**q).split(' ')
            retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=args.k)

            history_string = "".join(retrieved_history)
            prompt = history_string + "\n" + prompt
            full_prompt = history_string + "\n" + full_prompt

        if args.add_profile and format_flag == True:
            prompt = profile + "\n" + prompt
            full_prompt = profile + "\n" + full_prompt

        train_data.append(
            {
                "prompt": prompt,
                "full_prompt": full_prompt
            }
        )

    # print(train_data)

    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(generate_and_tokenize_prompt).shuffle()

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_arguments,
        data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)


    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    output_name = "./ckpt/{}/k{}-{}-{}-OPPU_LoRA-{}".format(args.task_name, args.k, args.task_name, model_name.split('/')[-1], i)
    model.save_pretrained(output_name)

    model.eval()
    model.config.use_cache = True  # silence the warnings. Please re-enable for inference!

    # test inference
    if args.add_profile:
        profile = test_profile[i]['output']

    if k > 0:
        visible_history_list = test_data[i]['profile']
        for p in visible_history_list:
            for key, value in p.items():
                p[key] = get_first_k_tokens(p[key], 368)

        history_list = [prompt_template[args.task_name]['retrieval_history'].format(**p) for p in visible_history_list]

        tokenized_corpus = [doc.split(" ") for doc in history_list]
        bm25 = BM25Okapi(tokenized_corpus)

    test_question_list = []
    question_id_list = []

    for q in test_data[i]['query']:

        if args.task_name == 'citation':
            test_question = q['input']
            test_article = extract_citation_title(test_question)
            option1, option2 = extract_option(test_question, 1), extract_option(test_question, 2)
            test_prompt = prompt_template[args.task_name]['prompt'].format(test_article, option1, option2)

        else:
            test_question = q['input']
            test_article = extract_article(test_question)
            test_prompt =  prompt_template[args.task_name]['prompt'].format(test_article)

        if k > 0:
            tokenized_query = prompt_template[args.task_name]['retrieval_query_wokey'].format(test_article).split(" ")
            retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=args.k)
        
            history_string = "".join(retrieved_history)
            test_prompt = history_string + "\n" + test_prompt

        if args.add_profile:
            test_prompt = profile + "\n" + test_prompt

        test_question_list.append(test_prompt)
        question_id_list.append(q['id'])

    test_batch_list = split_batch(test_question_list, 1)
    out_list = []

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_batch_list), total=len(test_batch_list)):
            # try:
            sentences = batch
            inputs = tokenizer(sentences, return_tensors="pt", padding=True, return_token_type_ids=False)
            inputs = inputs.to(model.device)

            with torch.autocast(device_type="cuda"):
                outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    top_k=10,
                    temperature=args.temperature,
                    top_p=0.9,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=200
                )

            out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            out_list += out_sentence
            # except:
            #     out_list += ['']
                
    for i in range(len(out_list)):
        output = out_list[i].replace(test_question_list[i], '')
        pred_all.append({
            "id": question_id_list[i],
            "output": output
            })
        
        print(output)

output_file = {
    'task': name2taskid[args.task_name],
    'golds': pred_all,
    'model': model_name,
}

if args.add_profile:
    with open('./output/{}/output-OPPU-k{}-{}-{}-profile.json'.format(args.k, args.task_name, args.task_name, model_name.split('/')[-1]), 'w') as f:
        json.dump(output_file, f, indent=4)
else:
    with open('./output/{}/output-OPPU-k{}-{}-{}.json'.format(args.k, args.task_name, args.task_name, model_name.split('/')[-1]), 'w') as f:
        json.dump(output_file, f, indent=4)