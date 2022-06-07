import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class simple_MLP(nn.Module):
    def __init__(self,dims):
        super(simple_MLP, self).__init__()
        modules = []
        # print(f'Making MLP with layers: {dims}')
        for dim1, dim2 in zip(dims[:-2],dims[1:-1]):
            modules.append(nn.Linear(dim1, dim2))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(dims[-2], dims[-1]))

        self.layers = nn.Sequential(*modules)
        
    def forward(self, x):
        if len(x.shape)==1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

class Tab_Transformer(nn.Module):
    def __init__(self, n_in, n_embed, atn_heads, mlpfory_layers, depth=1, viz_path=None):
        super().__init__()

        self.n_embed = n_embed
        self.atn_heads = atn_heads
        self.depth = depth

        # Embed categorical data and CLS token with Embedding (as in SAINT)
        self.cat_embeds = nn.Embedding(1, self.n_embed)

        self.embeds = nn.ModuleList([
            simple_MLP((1, self.n_embed)) for _ in range(n_in)
        ])
        
        # Conditional if needing to store attention weights for future visualization
        if viz_path:
             self.encoder_layer = TransformerEncoderLayerViz(self.n_embed, batch_first=True, nhead=self.atn_heads, \
                                                        dim_feedforward=self.n_embed*4, path=viz_path)
        else:
            self.encoder_layer = nn.TransformerEncoderLayer(self.n_embed, batch_first=True, nhead=self.atn_heads, \
                                                        dim_feedforward=self.n_embed*4)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, self.depth)

        self.mlpfory = simple_MLP(mlpfory_layers)

    # Not actually used in training loop (helps add clarity)
    def forward(self, x, cls):
        n1,n2 = x.shape
        x_embedded = torch.empty(n1, n2, self.n_embed)
        for i in range(n2):
            x_embedded[:,i,:] = self.embeds[i](x[:,i])
        # Adding embedded CLS Token to the start (convention to add to start?)
        cls_embedded = self.cat_embeds(cls)
        x_embedded = torch.cat((cls_embedded,x_embedded),dim=1)

        reps = self.transformer(x_embedded)
        # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
        y_reps = reps[:,0,:]
        output = self.mlpfory(y_reps)

        return output

# Code for storing attention weights for future visualization
# Based on official PyTorch code
class TransformerEncoderLayerViz(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, path, dropout = 0.1,
                 activation = F.relu,
                 layer_norm_eps = 1e-5, batch_first = False, norm_first = False,
                 device=None, dtype=None):
        self.path = path
        super(TransformerEncoderLayerViz, self).__init__(d_model, nhead, dim_feedforward, dropout,
                 activation, layer_norm_eps, batch_first, norm_first, device, dtype)
    
    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            val, attn_weights = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            np.save(os.path.join(self.path,'attention_weights.npy'),attn_weights.cpu().detach().numpy())
            x = x + val
            x = x + self._ff_block(self.norm2(x))
        else:
            val, attn_weights = self._sa_block(x, src_mask, src_key_padding_mask)
            np.save(os.path.join(self.path,'attention_weights.npy'),attn_weights.cpu().detach().numpy())
            x = self.norm1(x + val)
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(self, x,
                    attn_mask, key_padding_mask):
        x, attn_weights  = self.self_attn(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=True,
                            average_attn_weights=False)
        return self.dropout1(x), attn_weights