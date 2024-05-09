import os
from datasets import load_dataset
import torch
import json
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from lemma.lemma_utils.prefix_vars import PAD_TOKEN_ID, PAD_TOKEN, LEMMA_TOKEN, LEMMA_TOKEN_ID, SENTENCE_ENCODER_MAX_LENGTH
import spacy
from lemma.model.language_model.lemma_llama import LemmaLlama
from peft import PeftModel


LEMMA_TOKEN = "<|reserved_special_token_119|>"
SENTENCE_ENCODER_MAX_LENGTH = 4096
PAD_TOKEN = "<|reserved_special_token_112|>"
PAD_TOKEN_ID = 128117

LLAMA_SFT_TEMPLATE = (
"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}"
"<|eot_id|><|start_header_id|>context<|end_header_id|>\n\n{context}"
"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}"
"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
)

LEMMA_SFT_TEMPLATE = (
"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}"
"<|eot_id|><|start_header_id|>context-embedding<|end_header_id|>\n\n{context}"
"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}"
"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
)

nlp = spacy.load("en_core_web_sm")

def sentence_segmentation(text):
    doc = nlp(text)  # 텍스트 처리
    sentences = [sent.text.strip() for sent in doc.sents]  # 문장 추출
    return sentences

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k", \
                                                                    "llama3-8b", "lemma-llama"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--peft_path', type=str, default=None)
    return parser.parse_args(args)

@torch.inference_mode()
def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path, peft_path):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device, peft_path)
    prompt_structure = LEMMA_SFT_TEMPLATE if "lemma" in model_name else LLAMA_SFT_TEMPLATE

    for json_obj in tqdm(data):
        # This Part is for Dataset Cause Each Dataset has differenct Structure.
        if dataset in ['gov_report', 'multi_news', 'passage_count']:
            sys = prompt_format['system_prompt']
            context = json_obj['context']
            instruction = prompt_format['instruction']

        elif dataset in ['passage_retrieval_en']:
            sys = prompt_format['system_prompt']
            context = json_obj['context']
            instruction = prompt_format['instruction'][0] + json_obj['input'] + prompt_format['instruction'][1]

        else:
            sys = prompt_format['system_prompt']
            context = json_obj['context']
            instruction = prompt_format['instruction'] + json_obj['input']

        # For OUR LEMMA
        if "lemma" in model_name:
            input_sentences = sentence_segmentation(context)
            
            sent_input_ids = tokenizer(input_sentences, return_tensors='pt', padding=True)['input_ids'].to(device)
            ctx_features = model.encode(input_sentences_ids=sent_input_ids)

            context_embedding_len = len(ctx_features) if len(ctx_features) <= SENTENCE_ENCODER_MAX_LENGTH else SENTENCE_ENCODER_MAX_LENGTH
            context = LEMMA_TOKEN * context_embedding_len


        prompt_dict = {"system_prompt": sys, "context": context, "instruction": instruction}

        # TODO: Prompt에 맞게 다시 짜기
        prompt = prompt_structure.format(**prompt_dict)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)

        if "lemma" in model_name:
            input['ctx_features'] = ctx_features.unsqueeze(0)
            
        context_length = input.input_ids.shape[-1]

        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
    dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device, peft_path):
    if "llama3" in model_name:
        print(f"Loading **{model_name}** model")
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_cache=False,
            pad_token_id=PAD_TOKEN_ID,
            attn_implementation="flash_attention_2",
        ).to(device)
        tokenizer = LlamaTokenizer.from_pretrained(path)
        tokenizer.pad_token = PAD_TOKEN
        
    if "lemma" in model_name:
        print(f"Loading **{model_name}** model")
        
        model = LemmaLlama.from_pretrained(
            pretrained_model_name_or_path=peft_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_cache=False,
            pad_token_id=PAD_TOKEN_ID,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        tokenizer.pad_token = PAD_TOKEN


    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    peft_path = args.peft_path
    # define your model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
                    "passage_count", "passage_retrieval_en"]

    dataset2prompt = json.load(open("config/dataset2prompt_ver2.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            out_path = f"pred/{model_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
                        max_gen, prompt_format, dataset, device, model_name, model2path, out_path, peft_path))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
