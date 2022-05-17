from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, MBartForConditionalGeneration, MBart50Tokenizer, \
    XLMRobertaForCausalLM, XLMRobertaTokenizer
import torch
import json
import random
import numpy as np
import datasets
from datasets import load_metric, load_dataset
from torchtext.data.metrics import bleu_score

def get_src_tgt_text(file):
    src_text = []
    tgt_text = []

    for i in file:
        for conv in i["conversation"]:
            src_text.append(conv["en_sentence"])
            tgt_text.append(conv["ja_sentence"])
    
    return src_text, tgt_text

if __name__ == "__main__":

    torch.manual_seed(3233)
    random.seed(3233)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(3233)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(3233)

    # with open("/home/bpv991/Research Projects/multilingual_encoders/BSD/test.json") as f:
    #     test_json = json.load(f)
    
    # tgt_text, src_text = get_src_tgt_text(test_json)

    dataset = load_dataset("cfilt/iitb-english-hindi")
    print('load data complete')

    dataloader = torch.utils.data.DataLoader(dataset["test"], batch_size=16)
    print(len(dataloader))

    bleu = load_metric('bleu')

    # model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    model = XLMRobertaForCausalLM.from_pretrained("xlm-roberta-base")
    model.to("cuda")

    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    # tokenizer.src_lang = 'hi'
    # tokenizer.tgt_lang = 'hi'
    
    # print('load model complete')

    # inp_batches = np.array_split(src_text[:1000], 40)
    # tgt_batches = np.array_split(tgt_text[:1000], 40)

    print('start generate')
    bleu_score_tot = 0.0
    for idx, batch in enumerate(dataloader):
        ref = batch["translation"]["hi"]
        batch = batch["translation"]["en"]
        encoded_inp = tokenizer(batch, max_length = 50, padding = "max_length", truncation = True, return_tensors="pt")
        encoded_inp.to("cuda")
        generated_tokens = model.generate(**encoded_inp, forced_bos_token_id=tokenizer.bos_token_id)
        out_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        pred = []
        refs = []
        for idx, o in enumerate(out_text):
            samp = []
            pred.append(o.split(' '))
            samp.append(list(ref[idx].split(' ')))
            refs.append(samp)
        bleu_score_tot += bleu.compute(predictions = pred, references = refs)['bleu']
    
    print(bleu_score_tot)
    print(bleu_score_tot / len(dataloader))