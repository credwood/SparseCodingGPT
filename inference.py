"""
adapted from: https://github.com/zeyuyun1/TransformerVis/blob/main/inference.py
"""

import argparse
import numpy as np
import os
import sys

# import sparsify
import sparsify_PyTorch

import torch
from tqdm import tqdm
import scipy as sp
import sklearn
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from datasets import load_dataset

import random
import logging

from core import batch_up, example_dim_old, merge_two, get_inputs



def main():

    logging.basicConfig(filename="inference_log.log", encoding='utf-8', level=logging.DEBUG)
    model_version = args.model_version

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_version)
    model = HookedTransformer.from_pretrained(model_version)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Load data
    sentences = np.load(args.data_dir).tolist()[:args.num_instances]
    
    # building dictionaries to collect examples for each num_transformer_factors. 
    # notice that this "dictionary" is just the python dictionary, 
    # it's not the dictionary we used for dictionary learning.
    good_examples_contents = {}
    for d in range(args.num_transformer_factors):
        good_examples_contents[d] = []
      # good_examples_contents_new = {}
      # for d in range(args.num_transformer_factors):
            # good_examples_contents_new[d] = []

    # shard our data set into piece to fit into RAM
    sentences_shards = list(batch_up(sentences, batch_size=args.shard_size))
    logging.info("Numbers of sentences: {}".format(len(sentences)))

    # start the process to collect top activated example
    for shard_num in tqdm(range(len(sentences_shards)-1),'shards'):
       # define some parameters use for laters
        sentences_str = []
        words = []
        word_to_sentence = {} # word indice -> sentence (it belongs to) indice （sentence index, position in the sentence）
        sentence_to_word = {}
        n1 = 0
        n2 = 0
        X_set = {hook: [] for hook in args.hook_layer.keys()}
        # put our data into batch and ready to feed into transformer model
        sentences_batched = list(batch_up(sentences_shards[shard_num],batch_size=args.batch_size_1))
        for batch_idx in tqdm(range(len(sentences_batched)),'collect hidden states'):
            
            # this parts of the code looks complicated, 
            # but it basically keep track of a map between the word in each sentence 
            # to each of those sentences for convinience
            batch = sentences_batched[batch_idx]
            _, inputs_no_pad_ids = get_inputs(tokenizer, batch, device=device)
            pad_lens = [len(s) for s in inputs_no_pad_ids]

            for tokens in inputs_no_pad_ids:
                tokenized = [tokenizer.decode(token) for token in tokens] # `convert_ids_to_tokens` method for GPT has bug
                sentences_str.append(tokenized)
                words.extend(tokenized)
                w_index = []
                for j in range(len(tokenized)):
                    word_to_sentence[n2] = (n1,j)
                    w_index.append(n2)
                    n2+=1
                sentence_to_word[n1] = w_index
                n1+=1
                
            # Collect hidden_states of a particular layers from the Transformer model. We also concadenate the hidden states of each 
            # sentences (a sequence of vectors) into a giant list (we use this later for sparse code inferences).
            _ , hidden_states = model.run_with_cache(batch, prepend_bos=args.prepend_bos)
            
            for hook, layer in args.hook_layer.items():
                hook_hidden_states = None
                for name, t in hidden_states.items():
                    if int(name.split(".")[1]) == layer and hook == name.split(".")[-1]:
                        hook_hidden_states = t
                        break
                assert hook_hidden_states is not None, f"hook name: {hook} and layer: {layer} not in activation cache "
                
                for i in range(len(hook_hidden_states)):
                    sentences_trunc = hook_hidden_states[i][pad_lens[i]:]
                    for s in range(len(sentences_trunc)):
                        X_set[hook].append(sentences_trunc[s])

        # load dictionaries
        basis_dict = {hook: torch.from_numpy(np.load(path)).device() for hook, path in args.dictionary_dir.items()}
        assert list(basis_dict.keys()).sort() == list(args.num_transformer_factors.keys()).sort() == list(X_set.keys()).sort(), "dictionary_dir, num_transformer_factors and hook_layer arguments must all have exactly the same hook names as keys"
        
        # sparse code inference for each dictionary
        for hook, basis in basis_dict.items():
            # we batch the hidden states we collected from the last steps using a larget batch size
            X_set_batched = list(batch_up(X_set[hook], args.batch_size_2))
            X_sparse_set = []
            for i in tqdm(range(len(X_set_batched)),'sparse_inference'):
                batch = X_set_batched[i]
                I_cuda = torch.from_numpy(np.stack(batch, axis=1)).cuda()
                X_sparse = sparsify_PyTorch.FISTA(I_cuda, basis, args.reg, 500)[0].T
                X_sparse_set.extend(X_sparse.cpu().detach().numpy())
                
            # We save the top n activated examples for each transformer factor in a dictionary. An examples contains the following: The word that corresponds to the embedding vector, the context sentence, the position of the word int he context sentence, the level of activation.
            for d in range(args.num_transformer_factors[hook]):
                good_examples_contents[d] = merge_two(example_dim_old(X_sparse_set,d,words,word_to_sentence,sentences_str,n=args.top_n_activation),good_examples_contents[d])[:args.top_n_activation]
                
            # save the examples, which are in python dictionaries
            if not os.path.exists(args.outfile_dir):
                os.makedirs(args.outfile_dir)
            
            np.save(f"{args.outfile_dir}{hook}_example_l_{args.layer[hook]}.npy", good_examples_contents)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dictionary_dir', type=dict, default = {"hook_resid_post":'./dictionaries/hook_resid_post_example_dict_short.npy', "hook_attn_out":'./dictionaries/hook_attn_out_example_dict_short.npy'},help=
                        'Dictionary of hook names (keys) and file paths for trained dictionaries using train.py. The trained dictionary is a shape (hidden_state, dictionary_size) array saved as npy file.')
    
    parser.add_argument('--outfile_dir', type=str, default = './top_activate_examples/', help=
                        'Directory to save output files.')
        
    parser.add_argument('--data_dir', type=str,default ='./data/sentences.npy', help=
                        'The path of data (a list of string, each string is a sentence of sequence of text with fixed length). The data is generated using data_generate.py. Since we dont need much data for dictionary learning (we can put all data in RAM at once), we save the data in npy file.')
    
    parser.add_argument('--num_instances', type=int, default=298489, help='The number of sentences in our datasets. You can adjust this number to use a smaller datasets')

    parser.add_argument('--hook_layer', type=int, default={"hook_resid_post": 1, "hook_attn_out": 1}, help='Dict of hook names (keys) and a layer index for inference (values)')
    
    parser.add_argument('--batch_size_1', type=int, default=10, help=
                        'This is the batch size for inference of transformer model. Basically, how many seqeuence we shove into our model at once. This number shouldnt be big because inference of transformer model took lots of memory.')
    
    parser.add_argument('--batch_size_2', type=int, default=100, help=
                        'This is the batch size for sparse code inference. This number can be big, but a batch size too big wouldnt really increase the speed of sparse enforce. Since its basically just an one layer neural network. Theres not much parrallel computing.')
    
    parser.add_argument('--num_transformer_factors', type=dict, default={"hook_resid_post": 2000, "hook_attn_out": 2000}, help=
                        'The size of the dictionaries. Equivalent to the number of transformer factors. Keys are TransformerLens hook names.')
    
    parser.add_argument('--shard_size', type=int, default=1000, help=
                        'TLDR: Make this number small if you have a memory error. This is number that indicates how much data (hidden states) that fits in your RAM at once. Recall that we are calculating the top-activated examples, so we need to calculate the top-n activations over the sparse code of all word vector. This is a really large number. Thus, we split this calculating max process in shards.')
    
    parser.add_argument('--reg', type=float, default=0.3, help=
                        'The regularization factor for sparse coding. You should use the same one you used in training')
    
    parser.add_argument('--top_n_activation', type=int, default=500, help=
                        'This number indicates how many examples do we collect for each transformer factor. By default, we collect top 200 activated examples.')

    parser.add_argument('--model_version', type=str, default='gpt2', help='The model type.')    
    
    parser.add_argument('--prepend_bos', type=bool, default=False, help='Option for HookedTransformer to prepend bos. If you are using a tokenizew')    

    args = parser.parse_args()

    main()