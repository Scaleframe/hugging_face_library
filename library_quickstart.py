# # print("hello borld")
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
classifier


classifier("We are very happy to show you the 🤗 Transformers Library")

# [{'label': 'POSITIVE', 'score': 0.9997681379318237}]


classifier("We are slightly happy to show you the 🤗 Transformers Library")

# [{'label': 'POSITIVE', 'score': 0.9997638463973999}]

"""Wow the second sentence is 000004.29 less positive than the first 🤔"""
# 'just kidding, that is a confidence in it being positive, not the percentage of positivity 💡'. 


# ---------
# This downloads distilbert-base-uncased-fintetuned-sst-2-english

results = classifier(["We are very happy to to show you 🤗 Transformers library.", "We hope that you don't hate it."])
# returns list

for result in results:
    print(f"{result['label']}, with score: {round(result['score'], 4)}")
 
# POSITIVE, with score: 0.9997
# POSITIVE, with score: 0.7739


"""Using a specific model""" 

# If we want to use a model that is trained on a lot of French data, we use the model hub.

# Tags for "French" and "text-classification" gives back a suggestion for “nlptown/bert-base-multilingual-uncased-sentiment”. 

classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")
    # Now classifer can deal with English, French, Dutch, German, Italian, and Spanish!
    # You can change the model to a local path on your machine, and you can also pass a model object and its associated tokenizer. 
    # Note: Task summary tutorial for models and tasks. 


"""AutoTokenizer""" # will automatically download the tokenizer that works with a downloaded model. 
"""AutoModelForSequenceClassification""" # seems like a class we will use to download the model itself. 

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model_name, tokenizer=tokenizer)

    # Notes: Example scripts for finte tuning pre-trained model: 
            # https://huggingface.co/transformers/master/examples.html

            # Tutorial for sharing your model with the community: 
            # https://huggingface.co/transformers/master/model_sharing.html


"""Under the hood: pretrained models:"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


"""Using Tokeninzer"""

# Note: Several rules can be applied to different tokenizers. 
# More details at tokeninzer summary: 
# https://huggingface.co/transformers/master/tokenizer_summary.html

inputs = tokenizer("We are very happy to show you the 🤗 Transformers library.")

"""inputs"""
# {'input_ids': [101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 100, 19081, 3075, 1012, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

"""Attention Mask: from glossary"""

"""Attention mask""" # shows you what the model should attend to, in case of padding. 
# 1 for yes attend, 0 for no, this is just padding. 

"""<SIDEBAR>: More on tokens / inputs: (From Glossary, Attention Mask)"""

"""Token Type ID's:""" 

    # When joining multiple sequences [CLS] is classifier, [SEP] is separator tag.
    # [CLS] SEQUENCE_A [SEP] SEQUENCE_B [SEP]

"""Tokenizer joining multiple sentences as single sequence"""
from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

sequence_a = "Hugging Face is based in NYC"
sequence_b = "Where is Hugging Face based?"

encoded_dict = bert_tokenizer(sequence_a, sequence_b)
decoded = bert_tokenizer.decode(encoded_dict["input_ids"])


print(decoded)
# [CLS] HuggingFace is based in NYC [SEP] Where is HuggingFace based? [SEP]

# Some models, like BERT use token type ids to denote sequences. 
encoded_dict['token_type_ids']
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]


"""Back to overview <END SIDEBAR>"""

"""Passing sentence to tokenizer"""

pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it"],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

    # pt_batch:
    # {'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]),
    #  'input_ids': tensor([[  101,  2057,  2024,  2200,  3407,  2000,  2265,  2017,  1996,   100,
    #          19081,  3075,  1012,   102],
    #         [  101,  2057,  3246,  2017,  2123,  1005,  1056,  5223,  2009,   102,
    #              0,     0,     0,     0]])}


    # Padding will be applied on the side the model would expect and using the padding token the model was pretrained with. Attention mask is updated as well. 


"""Using a model"""

"""Get final activations (pre SoftMax)"""

pt_outputs = pt_model(**pt_batch)

     # In 🤗 Transformers, all outputs are tuples (with only one element potentially). 
        # Here, we get a tuple with just the final activations of the model.

    # pt_outputs: SequenceClassifierOutput(loss=None, logits=tensor([[-4.0833,  4.3364],
        # [-0.8004,  0.7992]], grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)


    # note: All 🤗 Transformer models return activations PRE final activation function since
        # the final activation function is often fused with the loss. (ie: SoftMax)

"""Get predicitons (post SoftMax)"""

import torch.nn.functional as F
pt_predictions = F.softmax(pt_outputs[0], dim=-1)

# pt_predictions: 
    # tensor([[2.2043e-04, 9.9978e-01],
    #         [1.6804e-01, 8.3196e-01]], grad_fn=<SoftmaxBackward>)

"""Add labels"""

import torch
pt_outputs = pt_model(**pt_batch, labels = torch.tensor([1,0]))

    # Loss data is added here:
    # SequenceClassifierOutput(loss=tensor(0.8919, grad_fn=<NllLossBackward>), logits=tensor([[-4.0833,  4.3364],
        # [-0.8004,  0.7992]], grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)

    # Note: output class gives you auto complete in an ide, and you can also index with string or int. 
        #  Pretty sweet!

"""Save Tokenizer and Model"""

save_directory = 'hf_models'
tokenizer = tokenizer
model = pt_model
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

    # This saves tokenizer, model, config files, and vocab to 
        # your save directory (hf_models in this repo).

