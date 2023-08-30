"""
from: https://github.com/zeyuyun1/TransformerVis/blob/main/generating_training_data.py
"""
import os
from datasets import load_dataset,load_from_disk,concatenate_datasets,DatasetDict
import numpy as np
from transformers import AutoTokenizer
import re
import random
from core import batch_up
import argparse

def main():
    random.seed(99)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_version, use_fast=True
    )
    dataset = load_dataset("wikitext",'wikitext-103-v1')
    articles = []
    article = ''
    for text in dataset['train']['text']:
    #     text = dataset['train'][i]['text']
        if re.match(r"^ = [^=]",text):
            articles.append(article)
            article = ''
        article = article + text
    articles_long = [ar for ar in articles if len(ar)>2000]
    sentences_sample = random.sample(articles_long,int(len(articles_long)))

    tokens = []
    for arr in sentences_sample:
        blocks = batch_up(arr,2000)
        for b in blocks:
            tokens.extend(tokenizer(b,add_special_tokens=False)['input_ids'])

    tokens_batch = batch_up(tokens,args.max_seq_length)

    sentences = []
    for batch in tokens_batch:
        sentences.append(tokenizer.decode(batch))
        if len(sentences)>args.num_instances:
            break

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    np.save(args.output_file, sentences)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu index')
    parser.add_argument('--max_seq_length', type=int, default=64, help='The length of each sentences in our dataset')
    parser.add_argument('--num_instances', type=int, default=300000, help='The number of sentences in our datasets.')
    parser.add_argument('--output_dir', type=str, default='./data', help='the directory for output')
    parser.add_argument('--output_file', type=str, default='./data/sentences.npy', help='the path for output')
    parser.add_argument('--model_version', type=str, default='gpt2', help='Only Hugging Face GPT models supported.')    

    args = parser.parse_args()
    main()