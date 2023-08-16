import argparse

import numpy as np
import torch
from transformers import AutoTokenizer
from .modeling_gpt2 import GPT2Model

import tqdm

from core import print_example_with_saliency

def main():
    model_version = args.model_version

    # load model
    model = GPT2Model.from_pretrained(model_version)
    tokenizer = AutoTokenizer.from_pretrained(model_version)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    folder = './top_activate_examples_w_color'

    example_list = np.load(args.example_dir,allow_pickle=True)[None][0]
    basis1 = torch.from_numpy(np.load(args.dictionary_dir)).to(device)

    with open(folder + '/l_{}.txt'.format(args.l),'w') as the_file:
        for sparse_dim in tqdm(range(len(example_list))):
            examples = example_list[sparse_dim][:args.top_n_activation]
            aa =  print_example_with_saliency(model,tokenizer,args.l,basis1,examples,sparse_dim,num_features =args.num_features,num_samples = args.num_samples,repeat = False,BATCH_SIZE_1=8,BATCH_SIZE_2=200,reg=args.reg,feature_selection = args.feature_selection)
            the_file.write(aa+'\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--l', type=int, default=10, help='which layer of transformer model we want to visualzize')
    parser.add_argument('--num_features', type=int, default=30, help='numbers of tokens that has color from LIME')
    parser.add_argument('--num_samples', type=int, default=2001, help='numbers of perturbed example we generated in using LIME')
    parser.add_argument('--feature_selection', type=str, default='auto', help='How to select the num_features tokens in LIME')
    parser.add_argument('--example_dir', type=str, default='./top_activate_examples/example_l_1.npy', help='The path of the top activated examples from a given layer')
    parser.add_argument('--reg', type=float, default=0.3, help=
                    'The regularization factor for sparse coding. You should use the same one you used in training')
    parser.add_argument('--top_n_activation', type=int, default=40, help=
                        'This number indicates how many examples do we collect for each transformer factor. By default, we collect top 200 activated examples.')
    parser.add_argument('--dictionary_dir', type=str, default = './dictionaries/example_dict_long.npy',help=
                        'This is path for the a trained dictionary using train.py. The trained dictionary is a shape (hidden_state,dictionary_size) array saved as npy file.')
    parser.add_argument('--model_version', type=str, default='gpt2', help='The GPT model type.')  

    args = parser.parse_args()

    main()