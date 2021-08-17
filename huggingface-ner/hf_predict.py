import os
import sys

from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification

from spacy import displacy

# load model and build a nlp pipeline
model = AutoModelForTokenClassification.from_pretrained('./fine-tuned-model/covid19_symp_model')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased') 
nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy='first')

def print_colored_text(text, egs):
    render_dict = {}
    for eg in egs:
        render_dict[eg['start']] = {'clr': '\033[44m', 'tp': 's'}
        render_dict[eg['end']] = {'clr': '\033[0m', 'tp': 'e'}
    rendered_text = []
    flag = False
    for i, t in enumerate(text):
        if i in render_dict and render_dict[i]['tp'] == 's':
            rendered_text.append(render_dict[i]['clr'])
            rendered_text.append(t)
            flag = True
        elif i in render_dict and render_dict[i]['tp'] == 'e':
            rendered_text.append(render_dict[i]['clr'])
            rendered_text.append(t)
            flag = False
        else:
            rendered_text.append(t)
    if flag: rendered_text.append('\033[0m')
    rendered_text = ''.join(rendered_text)
    print(rendered_text)


def predict(texts):
    for i,text in enumerate(texts):
        print('* sentence %s' % i)
        result = nlp(text)
        print_colored_text(text, result)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        dataset_fn = sys.argv[1]
    else: 
        dataset_fn = 'sample.txt'
    
    # comment the follwing if you have GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    texts = open(dataset_fn).readlines()
    predict(texts)
