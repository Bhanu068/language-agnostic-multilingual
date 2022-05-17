import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import M2M100Tokenizer


class Multilingual_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, txt_file, tokenizer, transform=None):

        self.multilingual_text = open(txt_file).read().split("\n")
        self.transform = transform
        self.tokenizer = tokenizer
        self.multiling_dict = []
        for t in self.multilingual_text:
            dict_lang = {}
            src_lang, tgt_lang, rand_src_lang, rand_tgt_lang, src_lang_id, tgt_lang_id = t.split("\t")
            src_lang = self.tokenizer(src_lang, padding='max_length', truncation=True, return_tensors = "pt", max_length = 100)
            tgt_lang = self.tokenizer(tgt_lang, padding='max_length', truncation=True, return_tensors = "pt", max_length = 100)
            rand_src_lang = self.tokenizer(rand_src_lang, padding='max_length', truncation=True, return_tensors = "pt", max_length = 100)
            rand_tgt_lang = self.tokenizer(rand_tgt_lang, padding='max_length', truncation=True, return_tensors = "pt", max_length = 100)
            dict_lang["src_lang"] = src_lang.input_ids
            dict_lang["src_lang_mask"] = src_lang.attention_mask
            dict_lang["tgt_lang"] = tgt_lang.input_ids
            dict_lang["tgt_lang_mask"] = tgt_lang.attention_mask
            dict_lang["rand_src_lang"] = rand_src_lang.input_ids
            dict_lang["rand_src_lang_mask"] = rand_src_lang.attention_mask
            dict_lang["rand_tgt_lang"] = rand_tgt_lang.input_ids
            dict_lang["rand_tgt_lang_mask"] = rand_tgt_lang.attention_mask
            dict_lang["src_lang_id"] = src_lang_id
            dict_lang["tgt_lang_id"] = tgt_lang_id
            self.multiling_dict.append(dict_lang)


    def __len__(self):
        return len(self.multiling_dict)

    def __getitem__(self, idx):

        out = self.multiling_dict[idx]

        if out["src_lang_id"] == "en":
            out["src_lang_id"] = 0
        if out["src_lang_id"] == "kk":
            out["src_lang_id"] = 1
        if out["src_lang_id"] == "ru":
            out["src_lang_id"] = 2
        if out["src_lang_id"] == "fr":
            out["src_lang_id"] = 3
        if out["src_lang_id"] == "de":
            out["src_lang_id"] = 4
        if out["src_lang_id"] == "lt":
            out["src_lang_id"] = 5
            
        if out["tgt_lang_id"] == "en":
            out["tgt_lang_id"] = 0
        if out["tgt_lang_id"] == "kk":
            out["tgt_lang_id"] = 1
        if out["tgt_lang_id"] == "ru":
            out["tgt_lang_id"] = 2
        if out["tgt_lang_id"] == "fr":
            out["tgt_lang_id"] = 3
        if out["tgt_lang_id"] == "de":
            out["tgt_lang_id"] = 4
        if out["tgt_lang_id"] == "lt":
            out["tgt_lang_id"] = 5

        if self.transform:
            sample = self.transform(sample)

        return out["src_lang"], out["src_lang_mask"], out["tgt_lang"], out["tgt_lang_mask"], out["rand_src_lang"], out["rand_src_lang_mask"], out["rand_tgt_lang"], out["rand_tgt_lang_mask"], out["src_lang_id"], out["tgt_lang_id"]

if __name__ == "__main__":

    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

    dataset = Multilingual_Dataset("/home/bhanuv/projects/multilingual_text.txt", tokenizer)

    dataloader = DataLoader(dataset, batch_size = 4)

    for d in dataloader:
        print(d)
        break