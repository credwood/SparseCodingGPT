"""
dictionary training
adapted from: https://github.com/zeyuyun1/TransformerVis/blob/main/train.py
"""
import os
import argparse
# import imageio
import numpy as np
import numpy.linalg as la

import logging
import torch
from tqdm import tqdm
import scipy as sp
import sklearn
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

from datasets import load_dataset
import nltk
from nltk.probability import FreqDist
from sklearn.datasets import load_digits

# import sparsify
import sparsify_PyTorch
from core import batch_up, get_inputs, collect_hidden_states, sparsify_batch, FISTA_optim_dict

def main():
    logging.basicConfig(filename=args.training_log, encoding='utf-8', level=logging.DEBUG)
    assert len(args.hooks) == len(list(args.PHI_NUM_DICT.keys())), "Number of phi numbers and hooks specified must match."
    save_directory = './dictionaries/'
    training_dicts = {hook: [f'./dictionaries/{args.model_version}_{args.name}_reg{args.reg}_d{args.PHI_NUM_DICT[hook]}_epoch{args.epoches}'] for hook in args.hook}
    model_version = args.model_version

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_version)
    model = HookedTransformer.from_pretrained(model_version)
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

    #initilize the training dictionaries and some variable used in dictionary learning for eahc hook type
    for hook in training_dicts.keys():
        training_dicts[hook].append(FISTA_optim_dict(args.HIDDEN_DIM, args.PHI_NUM_DICT[hook], device))
    
    # dicts for activation collection and batch word frequency data
    frequency_temp = {hook: [] for hook in args.hook}
    X_set_temp = {hook:[] for hook in args.hook}
        
    #or you can load a dictionary. You might want to do this if you are high way trough training a dictionary. And you want to keep training it.
    if args.load:
        for hook, path in args.load.items():
            print(f'loading {hook} from: {path}')
            PHI = torch.from_numpy(np.load(path)).to(device)
            training_dicts[hook]["PHI"] = PHI

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

                for lst in training_dicts.values():
                    file_name, hidden_dict = lst
                    np.save(file_name, hidden_dict["PHI"].cpu().detach().numpy())

            batch = sentences_batched[batch_idx]
            _, inputs_no_pad_ids = get_inputs(tokenizer, batch, device=device)
            max_len = max([len(s) for s in inputs_no_pad_ids])
            pad_lens = [max_len-len(s) for s in inputs_no_pad_ids]

            _, hidden_states = model.run_with_cache(batch, prepend_bos=args.prepend_bos)
            layers = {}
            for hook in X_set_temp.keys():
                
                # this looks silly but we won't assume that the dictionary will maintain order
                # instead we take the layer number in the parameter name 
                # names are formated: `blocks.{layer num}.{optional layernorm}.{hook_name}`
                if args.sparsify_every_layer:
                    hook_hidden_states = [(int(name.split(".")[1]), t) for name, t in hidden_states.items() if hook == name.split(".")[-1]]
                else:
                    assert args.sparsify_specific_layers is not None, "If not sparsifying eveyr layer, must provide dictionary with layer number for each hook"
                    to_sparsify = args.sparsify_specific_layers
                    hook_hidden_states = [(int(name.split(".")[1]), t) for name, t in hidden_states.items() if (int(name.split(".")[1]) in to_sparsify[hook] and hook == name.split(".")[-1])]
                hook_hidden_states.sort()
                layers = [num for num, _ in hook_hidden_states]
                layers[hook] = layers
                hook_hidden_states = [t for _, t in hook_hidden_states]
                X_set_temp[hook].extend(collect_hidden_states(hidden_states, pad_lens, layers))

            # TODO: refactor this token frequency count strategy!
            
            freq_layer = max([len(l) for l in layers.values()])
            for l in range(freq_layer):
                # update word/sentence tracker and frequency
                for tokens in inputs_no_pad_ids:
                    tokenized = [tokenizer.decode(token) for token in tokens] # `convert_ids_to_tokens` method for GPT has bug
                    for hook in layers.keys():
                        if l < len(layers[hook]):
                            frequency_temp[hook].extend([data_analysis[w] if w in data_analysis else 1 for w in tokenized])
                    
            #Step 2: once we collece enough hidden states, we train the dictionary.
            if batch_idx%5==0 and batch_idx>0:
                for hook, (path, hidden_dict) in training_dicts.items():
                    X_set_batched = list(batch_up(X_set_temp[hook], args.batch_size_2))
                    words_frequency_batched = list(batch_up(frequency_temp[hook], args.batch_size_2))
                    training_dicts[hook] = [path, sparsify_batch(words_frequency_batched, X_set_batched, device, args.reg, **hidden_dict)]
                    X_set_temp[hook] = []
                    frequency_temp[hook] = []
                
#               At this points, we finish exhuast all the hidden states we collect to update the dictionary. So we will dump all the hidden states vectors and jump back to step 1. We also print our some statistic for dictionary training so one can check how good their training are.
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
    
    parser.add_argument('--name', type=str, default='short', 
                        help='The name you want to have for your trained dictionary files (hook names will be added to name)')
    
    parser.add_argument('--model_version', type=str, default='gpt2', help='Only Hugging Face GPT models supported.')    
    
    parser.add_argument('--prepend_bos', type=bool, default=False, help='Option for HookedTransformer to prepend bos')    

    parser.add_argument('--default_bos', type=bool, default=False, help='Whether or not tokenizer type automatically prepends a bos token.')    

    args = parser.parse_args()

    main()