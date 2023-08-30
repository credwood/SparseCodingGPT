import argparse
import os

import numpy as np
import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

from tqdm import tqdm

from core import print_example_with_saliency

def main():
    model_version = args.model_version

    # load model
    model = HookedTransformer.from_pretrained(model_version)
    tokenizer = AutoTokenizer.from_pretrained(model_version)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    example_list = np.load(args.example_dir,allow_pickle=True)[None][0]
    basis1 = torch.from_numpy(np.load(args.dictionary_dir)).to(device)

    if not os.path.exists(args.outfile_dir):
        os.makedirs(args.outfile_dir)

    with open(f"{args.outfile_dir}{args.hook_name}_l_{args.l}.txt",'w') as the_file:
        for sparse_dim in tqdm(range(len(example_list))):
            examples = example_list[sparse_dim][:args.top_n_activation]
            aa =  print_example_with_saliency(model, tokenizer, args.l, args.hook_name,
                                              basis1, examples, sparse_dim, num_features=args.num_features,
                                              num_samples=args.num_samples, repeat=False,
                                              BATCH_SIZE_1=8, BATCH_SIZE_2=200, reg=args.reg,
                                              feature_selection=args.feature_selection,
                                              device=device)
            the_file.write(aa+'\n')
 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--l', type=int, default=10, help='which layer of transformer model we want to visualzize')
    
    parser.add_argument('--num_features', type=int, default=30, help='numbers of tokens that has color from LIME')
    
    parser.add_argument('--num_samples', type=int, default=2001, help='numbers of perturbed example we generated in using LIME')
    
    parser.add_argument('--feature_selection', type=str, default='auto', help='How to select the num_features tokens in LIME')
    
    parser.add_argument('--example_dir', type=str, default='./top_activated_examples/hook_resid_post_example_l_1.npy', help='The path of the top activated examples from a given layer for a given hook')
    
    parser.add_argument('--outfile_dir', type=str, default = './top_activated_examples_saliency/', help=
                        'Directory to save saliency maps.')
    
    parser.add_argument('--hook_name', type=str, default = 'hook_resid_post', help="name of hook")
    
    parser.add_argument('--reg', type=float, default=0.3, help=
                    'The regularization factor for sparse coding. You should use the same one you used in training')
    
    parser.add_argument('--top_n_activation', type=int, default=40, help=
                        'This number indicates how many examples do we collect for each transformer factor. By default, we collect top 200 activated examples.')
    
    parser.add_argument('--dictionary_dir', type=str, default = './dictionaries/hook_resid_post_gpt2_sparse_dict_reg0.3_d2000_epoch2.npy',help=
                        'This is path for the a trained dictionary using train.py. The trained dictionary is a shape (hidden_state,dictionary_size) array saved as npy file.')
    
    parser.add_argument('--model_version', type=str, default='gpt2', help='The GPT model type.')  

    args = parser.parse_args()

    main()