import torch
import logging
import csv
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from transformers import BartTokenizer, M2M100Tokenizer, MBart50Tokenizer
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import random
import pickle
import pandas as pd
from language_agnostic import MBart_MLP, M2M100_MLP
from dataset import Multilingual_Dataset
import os

def embed(sentence):
    inputs=tokenizer(sentence,padding='max_length',truncation=True,return_tensors="pt",max_length=128)
    with torch.no_grad():
        outputs = embed_model(**inputs.to(device))
    sentence_embedding = torch.index_select(outputs[0],1,torch.tensor([0]).to(device)).squeeze()
    return sentence_embedding

def cal_loss(model, batch, device = "cuda"):
    device = "cuda"
    src_lang, src_lang_mask, tgt_lang, tgt_lang_mask, rand_src_lang, rand_src_lang_mask, rand_tgt_lang, rand_tgt_lang_mask, src_lang_id, tgt_lang_id = batch  
    
    src_lang = src_lang.to(device)
    src_lang_mask = src_lang_mask.to(device)
    tgt_lang = tgt_lang.to(device)
    tgt_lang_mask = tgt_lang_mask.to(device)
    rand_src_lang = rand_src_lang.to(device)
    rand_src_lang_mask = rand_src_lang_mask.to(device)
    rand_tgt_lang = rand_tgt_lang.to(device)
    rand_tgt_lang_mask = rand_tgt_lang_mask.to(device)
    src_lang_id = src_lang_id.to(device)
    tgt_lang_id = tgt_lang_id.to(device)
    
    # src_lang, src_lang_mask, tgt_lang, tgt_lang_mask, rand_src_lang, rand_src_lang_mask, rand_tgt_lang, rand_tgt_lang_mask, 
    # src_lang_id, tgt_lang_id = torch.LongTensor(src_lang).to(device), torch.LongTensor(src_lang_mask).to(device), 
    # torch.LongTensor(tgt_lang).to(device), torch.LongTensor(tgt_lang_mask).to(device), 
    # torch.LongTensor(rand_src_lang).to(device), torch.LongTensor(rand_src_lang_mask).to(device), 
    # torch.LongTensor(rand_tgt_lang).to(device), torch.LongTensor(rand_tgt_lang_mask).to(device), 
    # src_lang_id.to(device), tgt_lang_id.to(device)
    
    # with torch.no_grad():
    src_emb, lang_emb_src, meaning_emb_src, lang_iden_src = model(src_lang, src_lang_mask)["last_hidden_state"]
    # print(lang_emb_src.shape, meaning_emb_src.shape)
    src_emb, lang_emb_src, meaning_emb_src = src_emb.reshape(src_emb.shape[0], -1), \
        lang_emb_src.reshape(src_emb.shape[0], -1), meaning_emb_src.reshape(src_emb.shape[0], -1)
    trg_emb, lang_emb_trg, meaning_emb_trg, lang_iden_trg = model(tgt_lang, tgt_lang_mask)["last_hidden_state"]
    trg_emb, lang_emb_trg, meaning_emb_trg = trg_emb.reshape(trg_emb.shape[0], -1), \
        lang_emb_trg.reshape(trg_emb.shape[0], -1), meaning_emb_trg.reshape(trg_emb.shape[0], -1)

    _, lang_emb_rand_src, meaning_emb_rand_src, lang_iden_rand_src = model(rand_src_lang, rand_src_lang_mask)["last_hidden_state"]
    # print(lang_emb_rand_src.shape, meaning_emb_rand_src.shape)
    lang_emb_rand_src, meaning_emb_rand_src = lang_emb_rand_src.reshape(lang_emb_rand_src.shape[0], -1), \
        meaning_emb_rand_src.reshape(meaning_emb_rand_src.shape[0], -1)
    _, lang_emb_rand_trg, meaning_emb_rand_trg, lang_iden_rand_trg = model(rand_tgt_lang, rand_tgt_lang_mask)["last_hidden_state"]
    lang_emb_rand_trg, meaning_emb_rand_trg = lang_emb_rand_trg.reshape(lang_emb_rand_trg.shape[0], -1), \
        meaning_emb_rand_trg.reshape(meaning_emb_rand_trg.shape[0], -1)


    y = torch.ones(len(src_lang)).to(device)

    # print(meaning_emb_src.shape)
    # print(meaning_emb_trg.shape)
    # print(meaning_emb_rand_trg.shape)
    # print(meaning_emb_rand_src.shape)
    loss_meaning = cos_fn_m(meaning_emb_src, meaning_emb_trg, y) + cos_fn_m(
        meaning_emb_trg, meaning_emb_rand_trg, -y) + cos_fn_m(
        meaning_emb_src, meaning_emb_rand_src, -y)

    loss_recov = mse_fn(lang_emb_src + meaning_emb_src, src_emb)+mse_fn(lang_emb_trg + meaning_emb_trg, trg_emb)
    loss_lang_iden = cross_fn(lang_iden_src, src_lang_id) + cross_fn(
        lang_iden_trg, tgt_lang_id)

    loss_lang_emb = [cos_fn(lang_emb_src, lang_emb_rand_src, y),cos_fn(lang_emb_trg, lang_emb_rand_trg, y),cos_fn(
        lang_emb_src, lang_emb_trg, -y)]

    return loss_meaning + loss_recov + loss_lang_iden + loss_lang_emb[0] + loss_lang_emb[1],loss_meaning,loss_recov,loss_lang_iden,loss_lang_emb

def eval(model, val_loader):

    # print("Validating")
    logger.info("Validating")
    model.eval()
    val_loss = 0

    for idx, batch in enumerate(val_loader):
        loss, loss_m, loss_r, loss_li, loss_le = cal_loss(model, batch)
        val_loss += loss.item()  
    return val_loss / len(val_loader)  

if __name__ == '__main__':
    
    os.environ["PYTHONPATH"] = "/home/bhanuv/projects/multilingual_agnostic"

    np.random.seed(9001)
    random.seed(9001)
    torch.manual_seed(9001)
    torch.cuda.manual_seed(9001)
    torch.cuda.manual_seed_all(9001)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logging.basicConfig(filename="logger/lang_meaning_dis_train.log",
                format='%(asctime)s %(message)s',
                datefmt="%Y-%m-%d %H:%M:%S",
                filemode='a')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda"

    tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M')
    # tokenizer = MBart50Tokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
    # model = MBart_MLP()
    model = M2M100_MLP()
    model.to(device)
    
    dataset = Multilingual_Dataset("/home/bhanuv/projects/multilingual_agnostic/multilingual_text.txt", tokenizer)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, lengths = [6005, 858, 1715])
    
    train_loader = DataLoader(train_set, batch_size = 128, num_workers = 4, shuffle = True, pin_memory = True)
    val_loader = DataLoader(val_set, batch_size = 128, num_workers = 4, shuffle = False, pin_memory = True)
    test_loader = DataLoader(test_set, batch_size = 128, num_workers = 4, shuffle = False, pin_memory = True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    mse_fn = nn.MSELoss()
    cos_fn = nn.CosineEmbeddingLoss()
    cos_fn_m = nn.CosineEmbeddingLoss()
    cross_fn = nn.CrossEntropyLoss()
    min_val_loss = np.Inf
    epochs_no_improve = 0
    ex_count=0
    PATH='/home/bhanuv/projects/multilingual_agnostic/vector_sequence_checkpoints/m2m100/best_val.pt'
    train_PATH='/home/bhanuv/projects/multilingual_agnostic/vector_sequence_checkpoints/m2m100/train.pt'

    epochs = 70
    result_columns = list()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_m = 0
        train_r = 0
        train_li = 0
        train_le = [0]*3

        # print("Epoch" + str(epoch+1))

        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss, loss_m, loss_r, loss_li, loss_le = cal_loss(model, batch)
            train_loss += loss.item()
            train_m += loss_m.item()
            train_r += loss_r.item()
            train_li += loss_li.item()
            train_le[0] +=loss_le[0].item()
            train_le[1] += loss_le[1].item()
            train_le[2] += loss_le[2].item()
            loss.backward()
            optimizer.step()

        # print("Train Loss: " + str(float(train_loss / len(train_loader))))
        logger.info("Train Loss: " + str(float(train_loss / len(train_loader))))
        torch.save(model.state_dict(), train_PATH)

        val_loss = eval(model, val_loader)

        # print("Val Loss: " + str(float(val_loss)))
        logger.info("Val Loss: " + str(float(val_loss)))

        if val_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), PATH)
        else:
            epochs_no_improve += 1
        
        if epoch > 3 and epochs_no_improve >= 10:
            # print("Early stop training")
            logger.info("Early stop training")
            break