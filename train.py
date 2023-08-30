"""
dictionary training
adapted from: https://github.com/zeyuyun1/TransformerVis/blob/main/train.py
"""
import argparse
import numpy as np
import logging
import os

import nltk
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

from core import batch_up, get_inputs, collect_hidden_states, sparsify_batch, FISTA_optim_dict

def main():
    logging.basicConfig(filename="training_log.log", encoding='utf-8', level=logging.DEBUG)
    assert len(args.hooks) == len(list(args.PHI_NUM_DICT.keys())), "Number of phi numbers and hooks specified must match."
    save_directory = './dictionaries/'
    training_dicts = {hook: 
                      [f'./dictionaries/{hook}_{args.model_version}_{args.name}_reg{args.reg}_d{args.PHI_NUM_DICT[hook]}_epoch{args.epoches}'] 
                      for hook in args.hooks
                      }
    model_version = args.model_version

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_version)
    model = HookedTransformer.from_pretrained(model_version)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # load data
    sentences = np.load(args.training_data).tolist()[:args.num_instances]
    
    print(f"Numbers of sentences: {len(sentences)}")
    # collect the frequency of each word in training data. 
    # words with higher freqeuncies receive smaller weights during the dictionary update.
    # the reason for doing this is explained in the appendix of original Yun et al paper.
    words = []
    for s in sentences:
        words.extend(tokenizer.tokenize(s))
    data_analysis = nltk.FreqDist(words)
    for w in data_analysis:
        data_analysis[w] = np.sqrt(data_analysis[w])

    # initilize the training dictionaries and 
    # some variables used in dictionary learning for each hook type
    for hook in training_dicts.keys():
        training_dicts[hook].append(FISTA_optim_dict(args.HIDDEN_DIM, args.PHI_NUM_DICT[hook], device))
    
    # dicts for batch activation and word frequency collection
    frequency_temp = {hook: [] for hook in args.hooks}
    X_set_temp = {hook:[] for hook in args.hooks}
        
    # Instead of using newly instantiated basis dicts
    # you can load basis 'checkpoints' for specified hooks 
    if args.load:
        for hook, path in args.load.items():
            print(f'loading {hook} from: {path}')
            PHI = torch.from_numpy(np.load(path)).to(device)
            training_dicts[hook]["PHI"] = PHI

    # starting the dictionary training loop, the training loop is divided into the following 2 steps:
    # 1. collect hidden states from transformer. once enough hidden states are collected, we jump to step 2.
    # 2. use the hidden states from step 1 to update the dictionary. 
    sentences_batched = list(batch_up(sentences,batch_size=args.batch_size_1))
    for epoch in range(args.epoches):
        print("Epoch: {}".format(epoch))
        
        #Step 1: collecting hidden states: 
        for batch_idx in tqdm(range(len(sentences_batched)),'main loop'):
            if batch_idx%100==0:
                #save dictionary checkpoint:
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)

                for lst in training_dicts.values():
                    file_name, hidden_dict = lst
                    np.save(file_name, hidden_dict["PHI"].cpu().detach().numpy())

            batch = sentences_batched[batch_idx]
            _, inputs_no_pad_ids = get_inputs(tokenizer, batch, device=device)
            pad_lens = [len(s) for s in inputs_no_pad_ids]

            _, hidden_states = model.run_with_cache(batch, prepend_bos=args.prepend_bos)
            layers = {}
            for hook in X_set_temp.keys():
                
                # names are formated: `blocks.{layer num}.{optional layernorm}.{hook}`
                if args.sparsify_every_layer:
                    hook_hidden_states = [(int(name.split(".")[1]), t) for name, t in hidden_states.items() 
                                          if hook == name.split(".")[-1]
                                          ]
                else:
                    assert args.sparsify_specific_layers is not None, "If not sparsifying every layer, must provide dictionary with layer numbers for each hook"
                    to_sparsify = args.sparsify_specific_layers
                    hook_hidden_states = [(int(name.split(".")[1]), t) for name, t in hidden_states.items() 
                                          if (int(name.split(".")[1]) in to_sparsify[hook] and hook == name.split(".")[-1])
                                          ]
                
                # this looks silly but we won't assume that the dictionary will maintain order
                # instead we sort by the layer numbers in the full hook name.
                hook_hidden_states.sort()
                layers[hook] = len([num for num, _ in hook_hidden_states])
                hook_hidden_states = [t for _, t in hook_hidden_states]
                X_set_temp[hook].extend(collect_hidden_states(hook_hidden_states, pad_lens))

            # TODO: refactor this token frequency count strategy!
            
            freq_layer = max([l for l in layers.values()])
            for l in range(freq_layer):
                # update word/sentence tracker and frequency
                for hook, hook_len in layers.items():
                    if l < hook_len:
                        for tokens in inputs_no_pad_ids:
                            tokenized = [tokenizer.decode(token) for token in tokens] # `convert_ids_to_tokens` method for GPT seems to have bug, not using it
                            frequency_temp[hook].extend([data_analysis[w] if w in data_analysis else 1 for w in tokenized])
                    
            # step 2: train the dictionary.
            if batch_idx%5==0 and batch_idx>0:
                for hook, (path, hidden_dict) in training_dicts.items():
                    X_set_batched = list(batch_up(X_set_temp[hook], args.batch_size_2))
                    words_frequency_batched = list(batch_up(frequency_temp[hook], args.batch_size_2))
                    training_dicts[hook] = [path, sparsify_batch(words_frequency_batched, X_set_batched, device, args.reg, **hidden_dict)]
                    X_set_temp[hook] = []
                    frequency_temp[hook] = []
                
                # logging statistic for dictionary training.
                for hook, (_, hidden_dict) in training_dicts.items():
                    logging.info(f"for hook {hook}: batch {batch_idx}, snr: {hidden_dict['snr']}, act1 max: {hidden_dict['ActL1'].max()}, act1 min: {hidden_dict['ActL1'].min()}")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for (filename_save, hidden_dict) in training_dicts.values():
        np.save(filename_save, hidden_dict["PHI"].cpu().detach().numpy())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_instances', type=int, default=298489, help=
                        'The number of sentences in our datasets. You can adjust this number to use a smaller datasets')
    
    parser.add_argument('--sparsify_every_layer', type=bool, default=True, help='If true, trains every layer, if false trains even indexed layers.')    

    parser.add_argument('--sparsify_specific_layers', type=dict, default=False, help='dict of hook names as keys and list of layer numbers (ints) to sparsify.')    

    parser.add_argument('--epoches', type=int, default=2, help=
                        'numbers of epochs to train dictionary')
    
    parser.add_argument('--hooks', type=list, default=["hook_resid_post", "hook_attn_out"], help='List of names of hooks for dictionary training. Names must correspond to TransformerLens hooks.')

    parser.add_argument('--PHI_NUM_DICT', type=dict, default={"hook_resid_post": 2000, "hook_attn_out": 2000}, help=
                        'The size of the dictionary. Also equivalent to the number of transformer factors. Keys are TransformerLens hook names.')
    
    parser.add_argument('--HIDDEN_DIM', type=int, default=768, help=
                        'The size of hidden state of your transformer model. The default the size of hidden states of gpt2')

    parser.add_argument('--batch_size_1', type=int, default=10, help=
                        'This is the batch size for inference of transformer model.')
    
    parser.add_argument('--batch_size_2', type=int, default=100, help=
                        'This is the batch size for sparse code inference. This number can be big, but a batch size too big wouldnt really increase the speed of sparse enforce. Since its basically just an one layer neural network. Theres not much parrallel computing.')
    
    parser.add_argument('--reg', type=float, default=0.3, help=
                        'The regularization factor for sparse coding. You should use the same one you used in inference ')
    
    parser.add_argument('--load', type=dict, default=None, help=
                        'Instead of intializing a random dictionary for training you can enter a dict of paths here with the hook names as keys and paths as values. The files must be a .npy file')
    
    parser.add_argument('--training_data', type=str, default='./data/sentences.npy', help=
                        'path of training data file. Again, must be a .npy file')
    
    parser.add_argument('--name', type=str, default='sparse_dict', 
                        help='The name for trained dictionary files (hook names will be prepended to full name)')
    
    parser.add_argument('--model_version', type=str, default='gpt2', help='Only Hugging Face GPT models supported.')    
    
    parser.add_argument('--prepend_bos', type=bool, default=False, help='Option for HookedTransformer to prepend bos. If tokenizer automatically prepends a bos this value must be set to True.')    


    args = parser.parse_args()

    main()