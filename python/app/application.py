import json
from flask import Flask, request, render_template, abort

from . import app
from . import main_logic as ml_service


from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
from sklearn.preprocessing import LabelEncoder

import torch.nn as nn
#%%
#test
# pretrained model from Transformers
MODEL_NAME = 'distilbert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
le = LabelEncoder.load('le.npy')
doc = pd.read_csv("doc_spec.py")
doc_names = pd.read_csv("doctor.py")

class DistilBertForSequenceClassification(nn.Module):
    def __init__(self, model_name, num_classes=None):
        super().__init__()

        config = AutoConfig.from_pretrained(
            model_name, num_labels=num_classes)

        self.distilbert = AutoModel.from_pretrained(model_name,
                                                    config=config)
        self.pre_classifier = nn.Linear(768,768)
        self.classifier = nn.Linear(768, 46)
        self.dropout = nn.Dropout(0.2)

    def forward(self, features, attention_mask=None, head_mask=None):
        assert attention_mask is not None, "attention mask is none"
        distilbert_output = self.distilbert(input_ids=features,
                                            attention_mask=attention_mask,
                                            head_mask=head_mask)
        # (bs, seq_len, dim)
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]                   # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)   # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)             # (bs, dim)
        pooled_output = self.dropout(pooled_output)          # (bs, dim)
        logits = self.classifier(pooled_output)              # (bs, dim)

        return logits


model = DistilBertForSequenceClassification(
    model_name=MODEL_NAME, num_classes=NUM_CLASSES)

model.load_state_dict(torch.load('doctor_model.pt', map_location=torch.device('cpu')))
model.eval()
#%%
@app.route("/")
def hello():
    return "Hello World!"


@app.route("/predict_doc", methods=['POST'])
def predict_doc():
    # data in string format and you have to parse into dictionary
    data = request.data
    dataDict = json.loads(data)

    tokenized = tokenizer(dataDict['text'], max_length=64,
                          padding="max_length", truncation=True, return_tensors="pt")
    predicted = model(tokenized['input_ids'], tokenized['attention_mask']), params = dict(model.named_parameters())
    predicted_t = le.transform(predicted)
    doc_cat = doc.loc[doc['specialty_id' == predicted_t]]
    doc_id = doc_cat['doc_id'].iloc[0]
    res = doc_names.loc[doc_names['id'] == doc_names][0]
    return res

# %%
