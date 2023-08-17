"""
adapted from: https://github.com/zeyuyun1/TransformerVis/blob/main/util.py
"""

import re
import numpy as np
import sys
import torch
from tqdm import tqdm
import scipy as sp
import sklearn
import torch.nn.functional as F
from IPython.display import HTML as html_print
from matplotlib import colors
import string
from math import log, e

# import sparsify
import sparsify_PyTorch

# import lime
from lime.lime_text import LimeTextExplainer

result = string.punctuation

def merge_two(list1, list2):
    ls = list1 + list2
    ls.sort(key = lambda x: x['score'],reverse=True)
    return ls

def batch_up(iterable, batch_size=1):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]

def example_dim_old(source, dim, words, word_to_sentence, sentences_str,
                    n=5, head=None, verbose = True, vis=False):
    my_df = []
    dim_slices = [x[dim] for x in source]
    indx = np.argsort(-np.array(dim_slices))[:n]
#     indx = np.argpartition(dim_slices,-n)[-n:]
    for i in indx:
        word = words[i]
        act = dim_slices[i]
        sent_position, word_position = word_to_sentence[i]
        sentence= sentences_str[sent_position]
        d = {'word':word,'index':word_position,'sent_index':sent_position,'score':act,'sent':"".join(sentence),}
        my_df.append(d)
    return my_df

def get_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    lens = [len(t) for t in token_lists]
    maxlen = max(lens)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    elif "<pad>" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("<pad>")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    ), token_lists


# Utility functions for collecting hidden states and applying optimization methods --- #

def collect_hidden_states(hidden_states, pad_lens, layers):
    """
    From a batch of hidden states, returns a list of vectors without padding dimensions

    Args:
        hidden_states: a batch of model hidden states, either be mid-layer or post layer
        pas_lens: lengths of each sentences the batch

    Returns:
        List of hidden state vectors without padding dimensions
    """
    X_set_temp = []
    for l in layers:
        X=hidden_states[l].cpu().detach().numpy()
        for i in range(len(X)):
            sentences_trunc = X[i][pad_lens[i]:] # padding is on left
            for s in range(len(sentences_trunc)):
                X_set_temp.append(sentences_trunc[s])
    
    return X_set_temp

def sparsify_batch(words_frequency_batched, hidden_batch, device, regularization, **kwargs):
    """
    Applies FISTA and basis update optimizations to a batch of hidden states
    
    Args:
        words_frequency_batched: a list of work frequencies, repeated for each layer for convenience
        hidden_batch: batch of hidden states
        device: training device
        kwargs: dictionary of optimization parameters, defined in the training loop
    
    returns: 
        dictionary of updated optimization paramters
    """
    PHI_SIZE = kwargs["PHI_SIZE"]
    PHI = kwargs["PHI"]
    lambd = kwargs["lambd"]
    ACT_HISTORY_LEN = kwargs["ACT_HISTORY_LEN"]
    HessianDiag = kwargs["HessianDiag"]
    ActL1 = kwargs["ActL1"]
    signalEnergy = kwargs["signalEnergy"]
    noiseEnergy = kwargs["noiseEnergy"]
    snr = kwargs["snr"]

    for i in tqdm(range(len(hidden_batch)),'update dictionary'):
        batch = hidden_batch[i]
        I_cuda = torch.from_numpy(np.stack(batch, axis=1)).to(device)
        frequency = torch.tensor(words_frequency_batched[i]).float().to(device)
        ahat, Res = sparsify_PyTorch.FISTA(I_cuda, PHI, regularization, 500)

        #Statistics Collection
        ActL1 = ActL1.mul((ACT_HISTORY_LEN-1.0)/ACT_HISTORY_LEN) + ahat.abs().mean(1)/ACT_HISTORY_LEN
        HessianDiag = HessianDiag.mul((ACT_HISTORY_LEN-1.0)/ACT_HISTORY_LEN) + torch.pow(ahat,2).mean(1)/ACT_HISTORY_LEN

        signalEnergy = signalEnergy*((ACT_HISTORY_LEN-1.0)/ACT_HISTORY_LEN) + torch.pow(I_cuda,2).sum()/ACT_HISTORY_LEN
        noiseEnergy = noiseEnergy*((ACT_HISTORY_LEN-1.0)/ACT_HISTORY_LEN) + torch.pow(Res,2).sum()/ACT_HISTORY_LEN
        snr = signalEnergy/noiseEnergy

        #Dictionary Update
        PHI = sparsify_PyTorch.quadraticBasisUpdate(PHI, Res*(1/frequency), ahat, 0.001, HessianDiag, 0.005)

    return {
        "PHI_SIZE": PHI_SIZE,
        "PHI": PHI,
        "lambd": lambd,
        "ACT_HISTORY_LEN": ACT_HISTORY_LEN,
        "HessianDiag": HessianDiag,
        "ActL1": ActL1,
        "signalEnergy": signalEnergy,
        "noiseEnergy": noiseEnergy,
        "snr": snr,
    }



    

# --- Utility class for getting intermediate hidden states --- #
class HookPoint(torch.nn.Module):
    # credit to Neel Nanda: https://colab.research.google.com/drive/1F6_1_cWXE5M7WocUcpQWp3v8z4b1jL20#scrollTo=zCw06jhGTEsH
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []
    
    def give_name(self, name):
        # Called by the model at initialisation
        self.name = name
    
    def add_hook(self, hook, dir='fwd'):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format (this includes input and output, 
        # which are the same for a HookPoint)
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)
        if dir=='fwd':
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir=='bwd':
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")
    
    def remove_hooks(self, dir='fwd'):
        if (dir=='fwd') or (dir=='both'):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir=='bwd') or (dir=='both'):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ['fwd', 'bwd', 'both']:
            raise ValueError(f"Invalid direction {dir}")
    
    def forward(self, x):
        return x

# ---- Utility methods for generating saliency maps --- #

def cstr(s, color='black'):
    return "<text style=color:{}>{}</text>".format(color, s)

def blend(alpha, color =[255,0,0], base=[255,255,255]):
    '''
    color should be a 3-element iterable,  elements in [0,255]
    alpha should be a float in [0,1]
    base should be a 3-element iterable, elements in [0,255] (defaults to white)
    '''
    out = [int(round((alpha * color[i]) + ((1 - alpha) * base[i])))/255 for i in range(3)]
    return colors.to_hex(out)

def cstr_color(s, opacity=0, switch= 0):
    if switch:
        return "<text style=background-color:{};>{}</text>".format(blend(opacity,color =[255,50,0]),s)
    else:
        return "<text style=background-color:{};>{}</text>".format(blend(opacity,color =[50,255,0]),s)

def decode_color(token,salient_map,word_index):
    """
    logic for dealing with # for BERT tokenization taken out
    """

    # dict ids
    temp = 0
    if word_index in salient_map:
        temp = salient_map[word_index]
    salient_map[word_index]=0
    max_weight = max(0.01,max(salient_map.values()))
    min_weight = min(-0.01,min(salient_map.values()))
    if temp<=max_weight:
        salient_map[word_index] = temp
    if temp>max_weight:
        salient_map[word_index] = max_weight

    max_weight = max(0.01,max(salient_map.values()))
    min_weight = min(-0.01,min(salient_map.values()))
    
    sent = ''
    for i in range(len(token)):
        w = token[i]
        if i==word_index:
            w= "<text style=color:{}>{}</text>".format('blue', w)
        if i in salient_map:
            if salient_map[i]>0:
                switch = 1
                opacity = salient_map[i]/max_weight
            else:
                switch = 0
                opacity = abs(salient_map[i])/abs(min_weight)
            w = cstr_color(w,opacity = opacity,switch = switch)

        sent += w

    return sent

def decode(token,ids):
    """
    logic for dealing with # for BERT tokenization taken out
    """
    sent = ''
    for i in range(len(token)):
        w = token[i]
        if i in ids:
            w = cstr(w,color ='blue')
        sent += w

    return sent

def generate_salient_map(model,tokenizer,l,basis1,text_instance,word_index,sparse_dim,
                         num_features,num_samples,BATCH_SIZE_1,BATCH_SIZE_2,reg,
                         feature_selection='auto'):
#     this function is modified from the LimeTextExplainer function from the lime repo:
#     https://github.com/marcotcr/lime/blob/a2c7a6fb70bce2e089cb146a31f483bf218875eb/lime/lime_text.py#L301
#     Zeyu Yun modified it to fit the huggingface style bert tokenizer.

    model_regressor=None
    explainer = LimeTextExplainer()
    encode = tokenizer(text_instance, add_special_tokens=False)['input_ids']

    inputs = [encode for i in range(num_samples)]

    distance_metric='cosine'


    def distance_fn(x):
        return sklearn.metrics.pairwise.pairwise_distances(
            x, x[0], metric=distance_metric).ravel() * 100
    
    def classifier_fn(inputs):

#         hook_1 = Save_int()
#         handle_1 = model.encoder.layer[l-1].attention.output.dropout.register_forward_hook(hook_1)
        #     inputs = tokenizer(str_to_predict,return_tensors='pt', add_special_tokens=False).cuda()
        I_cuda_ls = []
        inputs_batched = batch_up(inputs,BATCH_SIZE_1)
        for inputs in inputs_batched:
            inputs = torch.tensor(inputs).cuda()
            hidden_states = model(inputs,output_hidden_states=True).hidden_states # includes initial embedding layer
            X_att=hidden_states[l].cpu().detach().numpy()

            I_cuda_ls.extend(X_att[:,word_index,:])
        result= []
        I_cuda_batched = batch_up(I_cuda_ls,BATCH_SIZE_2)
        for batch in I_cuda_batched:
            I_cuda = torch.from_numpy(np.stack(batch, axis=1)).cuda()
            X_att_sparse = sparsify_PyTorch.FISTA(I_cuda, basis1, reg, 1000)[0].T
            result.extend(X_att_sparse[:,sparse_dim].cpu().detach().numpy())
    #     print(np.array(result).shape)

        return np.array(result).reshape(-1,1)

    doc_size = len(encode)
    sample = np.random.randint(1, doc_size + 1, num_samples - 1)
    data = np.ones((num_samples, doc_size))
    data[0] = np.ones(doc_size)
    features_range = range(doc_size)
    for i, size in enumerate(sample, start=1):
        inactive = np.random.choice(features_range, size,
                                            replace=False)
        data[i, inactive] = 0

    inverse_data= np.array(inputs)
    inverse_data[~data.astype('bool')]=tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
    distances = distance_fn(sp.sparse.csr_matrix(data))
    yss = classifier_fn(inverse_data)

    salient_map = dict(explainer.base.explain_instance_with_data(
                    data, yss, distances, 0, num_features,
                    model_regressor=model_regressor,
                    feature_selection=feature_selection)[1])
    
    return salient_map


def print_example_with_saliency(model,tokenizer,l,basis1,examples,sparse_dim,num_features =10,
                                num_samples = 1051,repeat = False,BATCH_SIZE_1=8,BATCH_SIZE_2=200,
                                reg=0.3,feature_selection='auto'):
    # text_instance = """music in the uk and ireland stating that the single" welds a killer falsetto chorus to a latterday incarnation of the' wall of sound'"."""
    final_print=''
    all_sentences={}
    for example in tqdm(examples):
        word_index = example['index']
        sent_index = example['sent_index']
        text_instance = example['sent']

        tokens = tokenizer.tokenize(text_instance)
#         if len(tokens)>70:
#             continue
        salient_map = generate_salient_map(model,tokenizer,l,basis1,text_instance,word_index,sparse_dim,num_features,num_samples,BATCH_SIZE_1,BATCH_SIZE_2,reg,feature_selection = feature_selection)
        result = decode_color(tokens,salient_map,word_index)
        if sent_index not in all_sentences:
            all_sentences[sent_index] = [result]
        else:
            if repeat:
                all_sentences[sent_index].append(result)
        #if we don't want repeated sentences
            else:
                continue
        if len(all_sentences)>20:
            break
    for ls in all_sentences.values():
        for block in ls:
            final_print = final_print + block + '<br />' + '<br />'
    return final_print
