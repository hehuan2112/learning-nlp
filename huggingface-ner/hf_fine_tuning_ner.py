from pathlib import Path
import re
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments

from datasets import load_metric

from spacy import displacy


def fine_tune(dataset_fn, 
        test_size=0.2, 
        model_name='distilbert-base-cased', 
        fine_tuned_model_path="./fine-tuned-model/covid19_symp_model",
        output_dir='./results'):

    def read_dataset(file_path):
        file_path = Path(file_path)

        raw_text = file_path.read_text().strip()
        raw_docs = re.split(r'\n\t?\n', raw_text)
        token_docs = []
        tag_docs = []
        for doc in raw_docs:
            tokens = []
            tags = []
            for line in doc.split('\n'):
                token, tag = line.split('\t')
                tokens.append(token)
                tags.append(tag)
            token_docs.append(tokens)
            tag_docs.append(tags)

        return token_docs, tag_docs


    def encode_tags(tags, encodings):
        labels = [[tag2id[tag] for tag in doc] for doc in tags]
        encoded_labels = []
        for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
            # create an empty array of -100
            doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
            arr_offset = np.array(doc_offset)

            # set labels whose first offset position is 0 and the second is not 0
            doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())

        return encoded_labels


    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # parse the texts and tags
    texts, tags = read_dataset(dataset_fn)

    # split into train val
    train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=test_size)

    # Next, let’s create encodings for our tokens and tags. 
    # For the tags, we can start by just create a simple mapping which we’ll use in a moment:
    unique_tags = set(tag for doc in tags for tag in doc)
    label_list = list(unique_tags)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    print('* train texts: ', len(train_texts))
    print('* unique tags: ', len(unique_tags))

    # get encodings
    train_encodings = tokenizer(train_texts, 
        is_split_into_words=True, return_offsets_mapping=True, 
        padding=True, truncation=True)
    val_encodings = tokenizer(val_texts, 
        is_split_into_words=True, return_offsets_mapping=True, 
        padding=True, truncation=True)

    # get labels
    train_labels = encode_tags(train_tags, train_encodings)
    val_labels = encode_tags(val_tags, val_encodings)

    # get datasets
    # we don't want to pass this to the model
    train_encodings.pop("offset_mapping") 
    val_encodings.pop("offset_mapping")

    train_dataset = MyDataset(train_encodings, train_labels)
    val_dataset = MyDataset(val_encodings, val_labels)

    training_args = TrainingArguments(
        output_dir=output_dir,           # output directory
        num_train_epochs=25,             # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, 
        num_labels=len(unique_tags), 
        id2label=id2tag, 
        label2id=tag2id
    )

    metric = load_metric("seqeval")
    return_entity_level_metrics = True
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        compute_metrics=compute_metrics
    )

    trainer.train()
    print(trainer.evaluate())
    model.save_pretrained(fine_tuned_model_path)
    print('* done fine-tuning and saved the fine-tuned model')

    return model, tokenizer


def predict(texts, model, tokenizer):
    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy='first')

    for i,text in enumerate(texts):
        print('* sentence %s' % i)
        result = nlp(text)
        print(result)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        dataset_fn = sys.argv[1]
    else:
        dataset_fn = 'dataset.tsv'
        
    # comment the follwing if you have GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
    # fine-tune!
    model, tokenizer = fine_tune(dataset_fn)
    
    # test the model with simple texts 
    texts = [
        "I was feeling bad all over about 7 pm aching and chills all over, and got nausea and extreme headache",
        "Moderna Covid-19 vaccine EUA Soreness at the injection site, body aches, chills, fatigue Treated effectively with ibuprofen",
        "Lightheaded immediately and almost passed out.  Felt a tingling in both hands.",
        "Metallic taste in mouth for about an hour.  8 hours later had the chills, body aches, headache, sinus pressure, and blocked ears.",
        "I feel head is ache and have some kind of breathing issue, but I don't take any aspirin"
    ]
    
    predict(texts, model, tokenizer)
