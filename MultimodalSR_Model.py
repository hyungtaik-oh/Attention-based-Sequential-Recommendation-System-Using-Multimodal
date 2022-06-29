'''
Building blocks for transformer's encoder in SASRec and actual pytorch lightning implementation
Author: bazman, JAN-2022
'''
# from torch.nn import MultiheadAttention, LayerNorm, Dropout, Conv1d, Embedding, BCEWithLogitsLoss
from torch.nn import LayerNorm, Dropout, Conv1d, Embedding, BCEWithLogitsLoss
import torch
import pytorch_lightning as pl
import numpy as np
from torch import nn 

from typing import Optional, Tuple
from torch.nn.parameter import Parameter
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
import torch.nn.functional as F

class VanillaAttention(nn.Module):
    """
    for gating operation
    
    Vanilla attention layer is implemented by linear layer.

    Args:
        input_tensor (torch.Tensor): the input of the attention layer

    Returns:
        hidden_states (torch.Tensor): the outputs of the attention layer
        weights (torch.Tensor): the attention weights
    
    """

    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.ReLU(True), nn.Linear(attn_dim, 1))

    def forward(self, input_tensor):
        # (B, Len, num, H) -> (B, Len, num, 1)
        energy = self.projection(input_tensor)
        weights = torch.softmax(energy.squeeze(-1), dim=-1)
        # (B, Len, num, H) * (B, Len, num, 1) -> (B, len, H)
        hidden_states = (input_tensor * weights.unsqueeze(-1)).sum(dim=-2)
        return hidden_states, weights

class PointWiseFF(torch.nn.Module):
    """
    Implementing point-wise feed-forward via conv1d
    Note it makes a transpose of last two dimentions inside a batch due to conv1d orientation
    """
    def __init__(self, d_model, dropout=0.2):
        super(PointWiseFF, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.dropout = torch.nn.Dropout(p=dropout)
    def forward(self, inputs):
        """
        g(x) = x + Dropout(g(LayerNorm(x)))
        where g(x) represents the self attention layer or the feed-forward network.
        That is to say, for layer g in each block, we apply layer normalization on the input x before feeding into g,
        apply dropout on g’s output, and add the input x to the final output.
        The dropout rate of turning off neurons is 0.2 for MovieLens-1m and 0.5 for the other three datasets due to their sparsity
        
        kernel size = 1 makes it a point-wise transformation
        remember, we have item embeddings in each column and kernel size corresponds to the number of columns that we aggregate in summation of convolution
        we transpose the inner matrix so we would have a column per embedding, not row
        this is due to conv1d - it mixes data between rows, i.e. samples
        """
        outputs = self.conv2(self.relu(self.conv1(inputs.transpose(-1, -2))))
        outputs = outputs.transpose(-1, -2) 
        outputs = self.dropout(outputs)
        
        return outputs

class MultiheadAttention(torch.nn.Module):
    __constants__ = ['batch_first']
    bias_k: Optional[torch.torch.Tensor]
    bias_v: Optional[torch.torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[torch.Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
    
    
class SASRecEncoderLayer(pl.LightningModule):
    """
    A building block for the Encoder
    Contains multi-head attention and point-wise feed-forward
    """
    def __init__(self, item_num, **kwargs):
        # item_num - number of items, i.e. vocab size
        super().__init__()
        # get params of a block as part of the object
        self.d_model = kwargs['d_model']
        self.num_heads = kwargs['num_heads']
        self.seq_len = kwargs['maxlen']
        self.dropout_rate = kwargs['dropout_rate']
        
        self.norm_1 = LayerNorm(self.d_model, eps=1e-8) # norm for self-attention
        self.norm_2 = LayerNorm(self.d_model, eps=1e-8) # norm for feed-forward
        self.attn= MultiheadAttention(embed_dim=self.d_model, 
                                       num_heads=self.num_heads, 
                                       dropout=self.dropout_rate, 
                                       batch_first=True) # single attention block
        
        self.ff = PointWiseFF(self.d_model, 
                              dropout=self.dropout_rate) # point-wise feed forward network
        
        # compile attention mask 
        # rows are target seq and columns are source seq
        # how to read the last line - last output item attends to all source/input items
        # how to read the first line - first output element attends ONLY to itself
        # when an element is True - it is excluded from attention
        # True in mask meaning that this element is excluded from attention
        # So we have a mask similar to this - lower left tiangular filled with False - those are pairs of active attention between target in rows and sources in columns
        # [False,  True,...  True,  True],
        # [False, False,...  True,  True],
        #               ...
        # [False, False,... False,  True],
        # [False, False,... False, False]]
        # self.attn_mask = torch.ones((self.seq_len, self.seq_len), dtype=torch.bool)
        # self.attn_mask = torch.triu(self.attn_mask, diagonal=1)
        self.register_buffer("attn_mask", torch.triu(torch.ones((self.seq_len, self.seq_len), 
                                                                dtype=torch.bool), 
                                                     diagonal=1))
    def forward(self, x_origin, x_fusion):  # *x는 x_origin
        # layer 2층 직접 만들음    
#         x_origin = x[:128]
#         x_fusion = x[128:]

        x_fusion = self.norm_1(x_fusion)
        x_origin = self.norm_1(x_origin) # norming before sending inside attention
        
        # x_2, _ = self.attn.forward(query=self.norm_1(x), 
#         print(x_fusion.size())
#         print(x_origin.size())
        x_2, _ = self.attn.forward(query=x_fusion, 
                                   key=x_fusion, 
                                   value=x_origin, 
                                   key_padding_mask=None, 
                                   need_weights=False, 
                                   attn_mask=self.attn_mask) # get multihead attention, second element in a return tuple is attention weights 
        x_origin = x_origin + x_2 # skip connection operation, dropout was inside attention block

        x_2 = self.norm_2(x_origin) # norm before sending to FF
        x_2 = self.ff(x_2) # dropout done inside FF
        x_origin = x_origin + x_2 # skip connection after FF
#         print('===================')
        
        ##### layer n+1번 추가
        x_origin = self.norm_1(x_origin)
        
        x_2, _ = self.attn.forward(query=x_fusion, 
                                   key=x_fusion, 
                                   value=x_origin, 
                                   key_padding_mask=None, 
                                   need_weights=False, 
                                   attn_mask=self.attn_mask) # get multihead attention, second element in a return tuple is attention weights
        x_origin = x_origin + x_2 # skip connection operation, dropout was inside attention block

        x_2 = self.norm_2(x_origin) # norm before sending to FF
        x_2 = self.ff(x_2) # dropout done inside FF
        x_origin = x_origin + x_2 # skip connection after FF
        
        return x_origin

class PositinalEncoder(pl.LightningModule):
    # makes a learnable positional encoding
    def __init__(self, seq_len, d_model):
        super(PositinalEncoder, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.pe = Embedding(num_embeddings=self.seq_len, embedding_dim=self.d_model)
    def forward(self, x):
        batch_size = x.shape[0]
        pe_for_one_sequence = self.pe(torch.arange(0,x.shape[1], dtype=torch.int).to(self.device)) # get a single positional embedding
        # positions = torch.tile(pe_for_one_sequence, (batch_size,1,1)).to(self.device) # replicate for batch # nice but is not supported by onnx
        positions = torch.cat(tuple(pe_for_one_sequence.unsqueeze(dim=0) for i in range(batch_size)), dim=0)
        return positions
    
    
class SASRecEncoder(pl.LightningModule):
    """
    Class for the Self-Attentive Sequential Recommendation Encoder https://arxiv.org/abs/1808.09781v1
    """
    @staticmethod
    def duplicate_blocks(module, N):
        # make copies of module N times
        from copy import deepcopy
        #return torch.nn.ModuleList([deepcopy(module) for i in range(N)])
        return torch.nn.Sequential(*[deepcopy(module) for i in range(N)])
    
    def __init__(self, item_num, fusion_type, **kwargs):
        '''
        - item_num - number of items in total for all data
        - warmup_proportion - in case lr-scheduler is used - pct of total iterations to warm-up linearly the lr
        - max_iters - in case lr-scheduler is used - max iters to zero lr
        - opt - optimizer to use
        - weight_decay - parameter for weight-decay
        - lr
        - d_model
        - num_blocks 
        - num_heads 
        - dropout_rate 
        - maxlen - max seq len
        - item_num - vocab size 
        - top_k - top_k for ndcg and hit rate calculation
        '''
        super().__init__()
        self.save_hyperparameters()
        self.ie = Embedding(num_embeddings=self.hparams.item_num+1, embedding_dim=self.hparams.d_model, padding_idx=0) # +1 in num_embeddings for padding id=0 -> 패딩아이템에 대한 인덱스 필요해서, 첫번째 아이템의 값은 패딩으로 0의 값을 가짐
        self.pe = PositinalEncoder(seq_len=self.hparams.maxlen, d_model=self.hparams.d_model)
        self.emb_dropout = Dropout(p=self.hparams.dropout_rate)
#         self.enc_stack = SASRecEncoder.duplicate_blocks(SASRecEncoderLayer(self.hparams.item_num, **kwargs), self.hparams.num_blocks)
        self.enc_stack = SASRecEncoderLayer(self.hparams.item_num, **kwargs)
        self.final_norm = LayerNorm(self.hparams.d_model, eps=1e-8)
        self.loss = BCEWithLogitsLoss()
                
        self.max_len = self.hparams.maxlen
        self.image_layer = torch.nn.Linear((4096), self.ie.weight.shape[1])
        self.text_layer = torch.nn.Linear((768), self.ie.weight.shape[1])
        
        self.fusion_type = fusion_type
        if self.fusion_type == 'concat' :
            self.fusion_layer = torch.nn.Linear((self.ie.weight.shape[1]*3), self.ie.weight.shape[1])
        
        elif self.fusion_type == 'gate' :
            self.fusion_layer = VanillaAttention(self.max_len,self.max_len)
        
        
    def forward(self, x, image_feature, text_feature):
        """
        x.dim = batch_size, seq_len
        out.dim - batch_size, seq_len, d_model
        """
        padding_mask = (x!=0).to(self.device) # True on real item positions and False on padding
        # produce embeddings from items
        d_model_sqrt = self.hparams.d_model**0.5 # from Attention is all you need paper
        ##### side info 추가 #####
        seq_latent = self.ie(x)
        seq_image = self.image_layer(image_feature.float())
        seq_text = self.text_layer(text_feature.float())
        
        if self.fusion_type == 'concat' :
            fusion_x = torch.cat((seq_latent, seq_image, seq_text), dim = -1)
            fusion_x = self.fusion_layer(fusion_x)
            
        elif self.fusion_type == 'sum' : 
            fusion_x = seq_latent + seq_image + seq_text
            
        elif self.fusion_type == 'gate' :
            fusion_x = torch.cat([seq_latent.unsqueeze(-2), seq_image.unsqueeze(-2), seq_text.unsqueeze(-2)], dim = -2)
            fusion_x,_ = self.fusion_layer(fusion_x)
        
        # fusion
        x_fusion = self.emb_dropout( fusion_x*d_model_sqrt + self.pe(x)) # add positional encoding and apply dropout to the embedding -> produce E-hat
#         x_emb = self.emb_dropout( self.ie(x)*d_model_sqrt + self.pe(x)) # add positional encoding and apply dropout to the embedding -> produce E-hat
        x_fusion = x_fusion*padding_mask.unsqueeze(-1)
    
        ### for NOVA input
        x_origin = self.emb_dropout(seq_latent*d_model_sqrt + self.pe(x))
        x_origin = x_origin*padding_mask.unsqueeze(-1)
#         x_origin = torch.cat((x_origin, x_fusion))
#         x = nn.ModuleList([x_origin, x_fusion]) # nn.sequential multiple input이 안되서 nn.ModuleList
        out = self.final_norm(self.enc_stack(x_origin, x_fusion)) # run through the stack of Encoder num_blocks

    
        return out
    
    def compute_relevance_scores(self, item_emb , q_items):
        """
        This is the prediction layer of SAS rec model
        It computes a dot product of embeddings ietm_emb obtainted through the transformer
        with q_items - candidates for the next item in a sequence
        item_emb is a (batch, seq, d_model) tensor 
        """
        q_emb = self.ie(q_items)
        # this is just batch dot product between candidate items and user sequence (item_emb)
        out = (item_emb*q_emb).sum(dim=-1) # dim = (batch, seq)
        return out
        
    def training_step(self, batch, batch_idx):
        """
        Perform a training step, given a batch
        Batch = u is user sequence(not used), seq is sequence of items as input, pos is seq shifted one item ahead, neg is sequence of items not in user selection

        """
        u, seq, pos, neg, image_feature, text_feature = batch
        
        item_emb = self.forward(seq, image_feature, text_feature) # get embeddings from transformer
        pos_scores = self.compute_relevance_scores(item_emb, pos) # scores for positive sequence
        neg_scores = self.compute_relevance_scores(item_emb, neg) # scores for negative sequence
        
        pos_labels = torch.ones(pos_scores.shape, device=self.device) 
        neg_labels = torch.zeros(neg_scores.shape, device=self.device)

        indices = torch.where(pos!=0) # exclude padding from loss computation
        
        # loss for positive and negative sequence
        loss = self.loss(pos_scores[indices], pos_labels[indices]) +\
        self.loss(neg_scores[indices], neg_labels[indices]) \
        + self.hparams.l2_pe_reg*torch.linalg.matrix_norm(next(self.pe.parameters()).data)
        # + self.hparams.l2_pe_reg*torch.linalg.matrix_norm(next(self.ie.parameters()).data) # regularization for item and positional embedding
        

        self.log('loss', loss.item(), prog_bar=True, logger=True)
        # self.log('lr', self.lr_scheduler.get_last_lr()[0],  prog_bar=True, logger=True)

        return {'loss': loss}
    
    def predict_step(self, items, items2score): # for inference
        '''
        Scores items2score as a candidates to continue items
        Input:
        items - known sequence of items
        items2score - item candidates to be next one in items sequence
        Returns:
        logits for each element from items2score
        '''
        with torch.no_grad:
            
            item_emb = self.forward(items) # shape (batch, seq_len, hidden_dim) = (batchx200x50)
            final_feat = item_emb[:, -1, :] # only use last embedding (batch, 50)
            
            # use shared embedding layer to get embeddings from items to score
            q_embs = self.ie(items2score) # shape (batch, seq_len, hidden_dim) = torch.Size([batch, 101, 50]) 

            # compute relevance scores via dot product
            logits = torch.bmm(q_embs, final_feat.unsqueeze(-1)).squeeze() # (batch, 101, 50)x(batch, 50,1)=(batch, 101, 1).squeeze() = (batch, 101)
        return logits # preds (batch, len(tems2score))
        
    def configure_optimizers(self):
        # param_optimizer = list(self.named_parameters())
        # no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

#         optimizer_grouped_parameters = [
#             {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
#              'weight_decay': 0.01},
#             {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
#              'weight_decay': 0.0}]

#         if self.hparams.opt == 'AdamW':
#             optimizer = torch.optim.AdamW(optimizer_grouped_parameters, 
#                                           lr=self.hparams.lr, 
#                                           weight_decay=self.hparams.weight_decay, 
#                                           betas=(0.9, 0.98))
        # select optimizer function
        if self.hparams.opt == 'FusedAdam':
            try:
                from apex.optimizers import FusedAdam
                selected_optimizer = FusedAdam
            except ModuleNotFoundError:
                print("\n No apex installed - switching to simple Adam\n")
                selected_optimizer = torch.optim.Adam
        elif self.hparams.opt == 'Adam':
            selected_optimizer = torch.optim.Adam
        elif self.hparams.opt == 'AdamW':
            selected_optimizer = torch.optim.AdamW

        # setup optimizer        
        opt = selected_optimizer(self.parameters(), 
                                       lr=self.hparams.lr, 
                                       weight_decay=self.hparams.weight_decay, 
                                       betas=(0.9, 0.98))


        # self.lr_scheduler = PolyWarmUpScheduler(optimizer,
        #                                         warmup=self.hparams.warmup_proportion,
        #                                         total_steps=self.hparams.max_iters)
        return opt
    
    # def optimizer_step(self, *args, **kwargs):
    #     super().optimizer_step(*args, **kwargs)
    #     # self.lr_scheduler.step()  # Step per iteration
    
    def _shared_val_step(self, batch, batch_idx):
        """
        Shared validation code for validation and for test datasets
        batch = 2, model dim = 50 as a sample data for dimentions
        """
#         final_seq, val_test_seq = batch
        final_seq, val_test_seq, image_feature, text_feature = batch

        with torch.no_grad():
            input_emb = self.forward(final_seq, image_feature, text_feature) # shape (batch, seq_len, hidden_dim) = (2x200x50)
            # take only last item embedding cos it containl linear combination of all items
            final_feat = input_emb[:, -1, :] # last hidden state/embedding, [2, 50] 
            
            # calculate embeddings from test_sequence
            val_test_emb = self.ie(val_test_seq) # shape [1, 101, 50])

            # item_embs.shape, final_feat.unsqueeze(-1).shape -> [2, 101, 50], [2, 50, 1]

            # get dot product of last hidden state with all embeddings
            logits = torch.bmm(val_test_emb, final_feat.unsqueeze(-1)) # [2, 101, 1]

            predictions = -logits.squeeze() # [2, 101]
            # in element with index 0 we have a logit for the ground truth item
            GROUND_TRUTH_IDX = 0

            TOP_N = self.hparams.top_k # number of items that we look for a proper recommendation in
            _, indices = torch.topk(predictions, TOP_N, dim=1, largest=False)
            _, rank = torch.where(indices == GROUND_TRUTH_IDX) # now we have ranks of ground truth elements #valid가 1개(실제 구매한 아이템), 구매하지 않은 아이템 100개
            HITS = torch.as_tensor(rank <= TOP_N , dtype=torch.int) # 0 for miss and 1 for hit
            NDCG = HITS/torch.log2(rank+2)
        return HITS.sum().item()/len(final_seq), NDCG.sum().item()/len(final_seq)
        
    def validation_step(self, batch, batch_idx):
        """
        calculate Hit Rate and NDCG on validation dataset
        """
        hits, ndcg = self._shared_val_step(batch, batch_idx)
        self.log('ndcg_val', ndcg, prog_bar=True, logger=True, sync_dist=True)
        self.log('hr_val', hits, prog_bar=True, logger=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        """
        calculate Hit Rate and NDCG on test dataset
        """
        hits, ndcg = self._shared_val_step(batch, batch_idx)
        self.log('ndcg_test', ndcg, prog_bar=True, logger=True, sync_dist=True)
        self.log('hr_test', hits, prog_bar=True, logger=True, sync_dist=True)