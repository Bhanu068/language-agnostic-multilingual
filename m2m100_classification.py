from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, MBartForConditionalGeneration, \
MBart50Tokenizer, MBartForSequenceClassification, MBartTokenizer
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, MBartForConditionalGeneration, \
MBart50Tokenizer, MBartForSequenceClassification, MBartTokenizer, M2M100Model, M2M100PreTrainedModel, M2M100Config, MBartConfig, \
XLMRobertaTokenizer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch
import torch.nn as nn
import random
import numpy as np
from datasets import load_metric, load_dataset
from torch.optim import AdamW
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from sklearn.metrics import classification_report
from language_agnostic import MLP

def get_src_tgt_text(file):
    src_text = []
    tgt_text = []

    for i in file:
        for conv in i["conversation"]:
            src_text.append(conv["en_sentence"])
            tgt_text.append(conv["ja_sentence"])
    
    return src_text, tgt_text

def preprocess_function(examples):
    return tokenizer(examples["hypothesis"], truncation=True)

def compute_metrics(eval_pred):
    # print(eval_pred)
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis = -1)
    print(classification_report(labels, predictions))
    return metric.compute(predictions=predictions, references=labels)

class MBartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

def _init_weights(config, module):
  std = 0.02
  if isinstance(module, nn.Linear):
      module.weight.data.normal_(mean=0.0, std=std)
      if module.bias is not None:
          module.bias.data.zero_()
  elif isinstance(module, nn.Embedding):
      module.weight.data.normal_(mean=0.0, std=std)
      if module.padding_idx is not None:
          module.weight.data[module.padding_idx].zero_()

# class M2M100(nn.Module):
#   def __init__(self, config: M2M100Config):
#     super().__init__()
#     self.model = MLP()
#     self.model.load_state_dict(torch.load("/home/bhanuv/projects/multilingual_agnostic/checkpoints/best_val.pt", map_location=torch.device('cpu')))
#     self.classification_head = MBartClassificationHead(
#                                   1024,
#                                   1024,
#                                   3,
#                                   0.1,
#                               )
#     _init_weights(config, self.classification_head.dense)
#     _init_weights(config, self.classification_head.out_proj)
  
#   def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         decoder_input_ids=None,
#         decoder_attention_mask=None,
#         head_mask=None,
#         decoder_head_mask=None,
#         cross_attn_head_mask=None,
#         encoder_outputs=None,
#         inputs_embeds=None,
#         decoder_inputs_embeds=None,
#         labels=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#     if labels is not None:
#             use_cache = False

#     if input_ids is None and inputs_embeds is not None:
#         raise NotImplementedError(
#             f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
#         )

#     _, _, hidden_states, _ = self.model(
#         x = input_ids,
#         attention_mask = attention_mask,
#     )
    
#     logits = self.classification_head(hidden_states)
#     loss = None
#     if labels is not None:
#         loss_fct = CrossEntropyLoss()
#         loss = loss_fct(logits.view(-1, 3), labels.view(-1))
#     if not return_dict:
#         output = (logits,)
#         return ((loss,) + output) if loss is not None else output

#     return Seq2SeqSequenceClassifierOutput(
#         loss=loss,
#         logits=logits,
#         past_key_values=None,
#         decoder_hidden_states=None,
#         decoder_attentions=None,
#         cross_attentions=None,
#         encoder_last_hidden_state=None,
#         encoder_hidden_states=None,
#         encoder_attentions=None,
#     )

class M2M100(nn.Module):
  def __init__(self, config: M2M100Config):
    super().__init__()
    self.model = M2M100Encoder.from_pretrained("facebook/m2m100_418M")
    self.classification_head = MBartClassificationHead(
                                  1024,
                                  1024,
                                  3,
                                  0.1,
                              )
    _init_weights(config, self.classification_head.dense)
    _init_weights(config, self.classification_head.out_proj)
  
  def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
    if labels is not None:
            use_cache = False

    if input_ids is None and inputs_embeds is not None:
        raise NotImplementedError(
            f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
        )

    outputs = self.model(
        input_ids = input_ids,
        attention_mask=attention_mask,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = outputs[0]  # last hidden state

    # eos_mask = input_ids.eq(2)

    # if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
    #     raise ValueError("All examples must have the same number of <eos> tokens.")
    # sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
    #     :, -1, :
    # ]
    logits = self.classification_head(hidden_states[:, 0, :])
    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 3), labels.view(-1))
    if not return_dict:
        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return Seq2SeqSequenceClassifierOutput(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        decoder_hidden_states=outputs.decoder_hidden_states,
        decoder_attentions=outputs.decoder_attentions,
        cross_attentions=outputs.cross_attentions,
        encoder_last_hidden_state=outputs.encoder_last_hidden_state,
        encoder_hidden_states=outputs.encoder_hidden_states,
        encoder_attentions=outputs.encoder_attentions,
    )


if __name__ == "__main__":

    torch.manual_seed(3233)
    random.seed(3233)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(3233)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(3233)

    en_dataset = load_dataset('xnli', language = 'zh', split = "test")

    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    # tokenizer.src_lang = 'en'

    model = M2M100(M2M100Config)
    model.load_state_dict(torch.load("/home/bhanuv/projects/multilingual_agnostic/classification_checkpoints/m2m100_0/checkpoint-1600/pytorch_model.bin"))

    for idx, m in enumerate(model.children()):
        if idx == 0:
            for param in m.parameters():
                if param.requires_grad:
                    param.requires_grad = False

    tokenized_dataset = en_dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    optimizer = AdamW(params = model.parameters(), lr = 2e-5)
    scheduler = StepLR(optimizer, step_size = 6000)

    metric = load_metric("accuracy")

    training_args = TrainingArguments(
    output_dir="/home/bhanuv/projects/classification_checkpoints/m2m100_0",
    learning_rate=2e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=80,
    weight_decay=0.01,
    logging_strategy = 'epoch',
    save_strategy = 'epoch',
    seed = 3233,
    fp16 = True,
    dataloader_num_workers=4,
    save_total_limit=1,
    evaluation_strategy = 'epoch',
    load_best_model_at_end = True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers = (optimizer, scheduler),
        compute_metrics = compute_metrics
    )

    # trainer.train()

    predictions = trainer.predict(tokenized_dataset)
    # print(np.argmax(predictions.predictions[0], axis = -1))