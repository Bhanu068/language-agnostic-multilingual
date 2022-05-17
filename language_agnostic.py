import torch
import torch.nn as nn
from transformers import M2M100Model, BartModel, MBartModel
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.lang_emb_layer = nn.Linear(1024, 1024)
        self.meaning_emb_layer = nn.Linear(1024, 1024)
        self.lang_iden_layer = nn.Linear(1024, 6)
    
    def avg_pool(self, input_ids, attention_mask):
        length = torch.sum(attention_mask, 1, keepdim=True).float()
        attention_mask = attention_mask.unsqueeze(2)
        hidden = input_ids.masked_fill(attention_mask == 0, 0.0)
        avg_input_ids = torch.sum(hidden, 1) / length
        return avg_input_ids

    def forward(self, x, attention_mask):
        lang_emb = self.lang_emb_layer(x)
        meaning_emb = self.meaning_emb_layer(x)
        lang_emb_pool = self.avg_pool(lang_emb, attention_mask)
        lang_iden = self.lang_iden_layer(lang_emb_pool)
        return x, lang_emb, meaning_emb, lang_iden

class M2M100_MLP(nn.Module):

    def __init__(self):
        super(M2M100_MLP, self).__init__()
        self.m2m100_encoder = M2M100Encoder.from_pretrained("facebook/m2m100_418M")
        self.m2m100_encoder.eval()
        self.mlp = MLP()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):
        with torch.no_grad():
            encoder_output = self.m2m100_encoder(input_ids = input_ids, attention_mask = attention_mask.reshape(attention_mask.shape[0], -1))
        
        encoder_output["last_hidden_state"], lang_emb, meaning_emb, lang_iden = self.mlp(encoder_output["last_hidden_state"],
                                                                                         attention_mask.reshape(attention_mask.shape[0], -1))
        encoder_output["last_hidden_state"] = (encoder_output["last_hidden_state"], lang_emb, meaning_emb, lang_iden)
        return encoder_output
    
class MBart_MLP(nn.Module):

    def __init__(self):
        super(MBart_MLP, self).__init__()
        self.mbartencoder = MBartModel.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.mbartencoder.eval()
        self.lang_emb_layer = nn.Linear(1024, 1024)
        self.meaning_emb_layer = nn.Linear(1024, 1024)
        self.lang_iden_layer = nn.Linear(1024, 6)

    def forward(self, x, attention_mask):
        with torch.no_grad():
            x = self.mbartencoder.encoder(input_ids = x, attention_mask = attention_mask.reshape(attention_mask.shape[0], -1)).last_hidden_state
            x = x[:, 0, :]
        lang_emb = self.lang_emb_layer(x)
        meaning_emb = self.meaning_emb_layer(x)
        lang_iden = self.lang_iden_layer(lang_emb)
        return x, lang_emb, meaning_emb, lang_iden

if __name__ == "__main__":

    model = MBart_MLP()
    print(model(torch.randint(0, 100, (1, 256)), torch.ones((1,1, 256)))[0].shape)