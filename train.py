"""
from: https://github.com/zeyuyun1/TransformerVis/blob/main/train.py

"""
import os
import argparse
# import imageio
import numpy as np
import numpy.linalg as la
import scipy.io
import sys
"""
dictionary training
adapted from: https://github.com/zeyuyun1/TransformerVis/blob/main/train.py
"""
import logging
import torch
from tqdm import tqdm
import scipy as sp
import sklearn
from transformers import AutoTokenizer
from modeling_gpt2 import GPT2Model

from datasets import load_dataset
import nltk
from nltk.probability import FreqDist
from sklearn.datasets import load_digits

# import sparsify
import sparsify_PyTorch
from core import batch_up, get_inputs, collect_hidden_states, sparsify_batch

def main():
    save_directory = './dictionaries/'
    filename_save = '''./dictionaries/{}_{}_reg{}_d{}_epoch{}'''.format(args.model_version,args.name,args.reg,args.PHI_NUM,args.epoches)
    attn_filename_save = '''./dictionaries/attn_{}_{}_reg{}_d{}_epoch{}'''.format(args.model_version,args.name,args.reg,args.PHI_NUM,args.epoches)

    model_version = args.model_version

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_version)
    model = GPT2Model.from_pretrained(model_version)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # load data
    sentences = np.load(args.training_data).tolist()[:args.num_instances]
    
    print("Numbers of sentences: {}".format(len(sentences)))
    # collect the frequency of each word in our training data. The word with high freqeuncy should receive a smaller weight
    #during the dictionary update. We took care of this in our training loop. The reason for doing this is explained in       the appendix 
    words = []
    for s in sentences:
        words.extend(tokenizer.tokenize(s))
    data_analysis = nltk.FreqDist(words)
    for w in data_analysis:
        data_analysis[w] = np.sqrt(data_analysis[w])

    #initilize the dictionary matrix and some variable used in dictionary learning
    PHI_ = torch.randn([args.HIDDEN_DIM, args.PHI_NUM]).to(device)
    PHI_ = PHI_.div_(PHI_.norm(2,0))
    hidden_dict = {
        "PHI_SIZE": [args.HIDDEN_DIM, args.PHI_NUM],
        "PHI": PHI_,
        "lambd": 1.0,
        "ACT_HISTORY_LEN": 300,
        "HessianDiag": torch.zeros(args.PHI_NUM).to(device),
        "ActL1": torch.zeros(args.PHI_NUM).to(device),
        "signalEnergy": 0.,
        "noiseEnergy": 0.,
        "snr": 1.,
    }

    # variables for attention dictionary
    PHI_ATTN_ = torch.randn([args.HIDDEN_DIM, args.PHI_NUM_ATTN]).to(device)
    PHI_ATTN_ = PHI_ATTN_.div_(PHI_ATTN_.norm(2,0))
    attn_hidden_dict = {
        "PHI_SIZE": [args.HIDDEN_DIM, args.PHI_NUM_ATTN],
        "PHI": PHI_ATTN_,
        "lambd": 1.0,
        "ACT_HISTORY_LEN": 300,
        "HessianDiag": torch.zeros(args.PHI_NUM_ATTN).to(device),
        "ActL1": torch.zeros(args.PHI_NUM_ATTN).to(device),
        "signalEnergy": 0.,
        "noiseEnergy": 0.,
        "snr": 1.,
    }
    
    frequency_temp = []
        
    #or you can also load a dictionary. You might want to do this if you are high way trough training a dictionary. And you want to keep training it.
    if args.load:
        print('load from: '+ args.load)
        PHI = torch.from_numpy(np.load(args.load)).to(device)
        hidden_dict["PHI"] = PHI

    #starting the dictionary training loop, the training loop is divided into the following 2 steps:
    #1. collect hidden states from transformer. Once we collect enough those hidden state vector, we jump to step 2.
    #2. Use the hidden state vectors collect from step 1 to update the dictionary. Once we are done with exhuast those hidden states. We jump back step 1 to collect more of those hidden states.
    sentences_batched = list(batch_up(sentences,batch_size=args.batch_size_1))
    for epoch in range(args.epoches):
        print("Epoch: {}".format(epoch))
        
        #Step 1: collecting hidden states using different input sentences from transformer model: 
        for batch_idx in tqdm(range(len(sentences_batched)),'main loop'):
            if batch_idx%100==0:
                #save your dictionary every now and then to avoid the unexpected crash during training loop:
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)

                np.save(filename_save, hidden_dict["PHI"].cpu().detach().numpy())

                if args.train_attention_dicts:
                    np.save(attn_filename_save, attn_hidden_dict["PHI"].cpu().detach().numpy())


            batch = sentences_batched[batch_idx]
            inputs, inputs_no_pad_ids = get_inputs(tokenizer, batch, device=device)
            max_len = max([len(s) for s in inputs_no_pad_ids])
            pad_lens = [max_len-len(s) for s in inputs_no_pad_ids]

            if args.train_attention_dicts:
                model.reset_hook_cache()
                model.remove_all_hooks()
                model.cache_all_hooks()

            hidden_states = model(**inputs,output_hidden_states=True).hidden_states # includes initial embedding layer
            layers = [num for num in range(len(hidden_states))]  if args.sparsify_every_layer else [num for num in range(len(hidden_states)) if not num%2]
            
            X_set_temp = collect_hidden_states(hidden_states, pad_lens, layers)
            if args.train_attention_dicts:
                attn_hidden_states = model.get_hook_cache()
                attn_layers = [num for num in range(len(attn_hidden_states.values()))]
                attn_hidden_states = list(attn_hidden_states.values())
                attn_X_set_temp = collect_hidden_states(attn_hidden_states, pad_lens, attn_layers)

            for l in layers:
                # update word/sentence tracker and frequency
                for tokens in inputs_no_pad_ids:
                    tokenized = [tokenizer.decode(token) for token in tokens] # `convert_ids_to_tokens` method for GPT has bug
                    frequency_temp.extend([data_analysis[w] if w in data_analysis else 1 for w in tokenized])
                
            #Step 2: once we collece enough hidden states, we train the dictionary.
            if batch_idx%5==0 and batch_idx>0:
                X_set_batched = list(batch_up(X_set_temp,args.batch_size_2))
                words_frequency_batched = list(batch_up(frequency_temp,args.batch_size_2))
                hidden_dict = sparsify_batch(words_frequency_batched, X_set_batched, device, args.reg, **hidden_dict)
                X_set_temp = []

                if args.train_attention_dicts:
                    attn_X_set_batched = list(batch_up(attn_X_set_temp,args.batch_size_2))
                    attn_hidden_dict = sparsify_batch(words_frequency_batched, attn_X_set_batched, device, args.reg, **attn_hidden_dict)
                    attn_X_set_temp = []

                frequency_temp=[]
                
#               At this points, we finish exhuast all the hidden states we collect to update the dictionary. So we will dump all the hidden states vectors and jump back to step 1. We also print our some statistic for dictionary training so one can check how good their training are.
                print(f"Total_step {epoch}, snr: {hidden_dict['snr']}, act1 max: {hidden_dict['ActL1'].max()}, act1 min: {hidden_dict['ActL1'].min()}")
                if args.train_attention_dicts:
                    print(f"attn dict: Total_step {epoch}, snr: {attn_hidden_dict['snr']}, act1 max: {attn_hidden_dict['ActL1'].max()}, act1 min: {attn_hidden_dict['ActL1'].min()}")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(filename_save, hidden_dict["PHI"].cpu().detach().numpy())
    if args.train_attention_dicts:
        np.save(attn_filename_save, attn_hidden_dict["PHI"].cpu().detach().numpy())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_instances', type=int, default=298489, help=
                        'The number of sentences in our datasets. You can adjust this number to use a smaller datasets')
    
    parser.add_argument('--sparsify_every_layer', type=bool, default=True, help='If true, trains every layer, if false trains even indexed layers.')    

    parser.add_argument('--epoches', type=int, default=2, help=
                        'numbers of epoch you want to train your dictionary')
    
    parser.add_argument('--PHI_NUM', type=int, default=2000, help=
                        'The size of the dictionary. Also equivalent to the number of transformer factors.')
    
    parser.add_argument('--PHI_NUM_ATTN', type=int, default=2000, help=
                        'The size of the attention dictionary. Also equivalent to the number of attention transformer factors.')
    
    parser.add_argument('--HIDDEN_DIM', type=int, default=768, help=
                        'The size of hidden state of your transformer model. The default the size of hidden states of gpt2')

    parser.add_argument('--batch_size_1', type=int, default=10, help=
                        'This is the batch size for inference of transformer model.')
    
    parser.add_argument('--batch_size_2', type=int, default=100, help=
                        'This is the batch size for sparse code inference. This number can be big, but a batch size too big wouldnt really increase the speed of sparse enforce. Since its basically just an one layer neural network. Theres not much parrallel computing.')
    
    parser.add_argument('--reg', type=float, default=0.3, help=
                        'The regularization factor for sparse coding. You should use the same one you used in inference ')
    
    parser.add_argument('--load', type=str, default=None, help=
                        'Instead of intialize an random dictionary for training. You can also enter a path here indicating the the path of the dictionary you want to start with. The file must be a .npy file')
    
    parser.add_argument('--training_data', type=str, default='./data/sentences.npy', help=
                        'path of training data file. Again, must be a .npy file')
    
    parser.add_argument('--name', type=str, default='short', 
                        help='The name you want to have for your trained dictionary file  ')

    parser.add_argument('--model_version', type=str, default='gpt2', help='Only Hugging Face GPT models supported.')    
    
    parser.add_argument('--train_attention_dicts', type=bool, default=True, help='Train basis for hidden state directly after attention layer. Defaults to True.')    

    args = parser.parse_args()

    main()