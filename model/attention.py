
import time
import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.functional import interpolate



class ProbAttention(nn.Module):
    def __init__(self,args ):
        super().__init__()
        self.factor = args.factor
        self.d_model = args.d_model
        self.args=args
        self.n_heads=args.n_heads
    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K,(L_Q, sample_k))  # 选择随机L_K列，index放在index_sample中# real U = U_part(factor*ln(L_k))*L_q
        #随机采样K列
        if self.args.useSample_k==True:
            K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        else:
            K_sample=K_expand
        x1 = Q.unsqueeze(-2)
        x2 = K_sample.transpose(-2, -1)
        Q_K_sample = torch.matmul(x1, x2).squeeze(-2)
        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), Q_K_sample.shape[-1])#！！！！L_k应该换成Q_K_sample最后维度数
        M_top = M.topk(n_top, sorted=False)[1] #n_top是选择最好的几个Q
        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k
        return Q_K, M_top
    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        V_sum = V.mean(dim=-2)
        contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        return contex
    def _update_context(self, context_in, V, scores, index):
        B, H, L_V, D = V.shape
        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)
        context_in[torch.arange(B)[:, None, None],torch.arange(H)[None, :, None],index, :] = torch.matmul(attn, V).type_as(context_in)
        return context_in
    def forward(self, queries, keys, values):
        B, L_Q, H, D = queries.shape  #B批量, L_Q Q长度, H 多头长度, D  维度
        _, L_K, _, _ = keys.shape     #L_K K长度
        queries = queries.transpose(2, 1)  #B,H,LQ,D
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        #U_part 选择K随机列的个数
        # 选最好的n_top个Q###scores_top=Q_K, index=M_top
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        # add scale factor
        scale = 1. / math.sqrt(D)
        #降低维度D的影响
        scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)#对values的-2维度均值
        # update the context with selected top_k queries
        context = self._update_context(context, values, scores_top, index)
        context=context.transpose(2, 1).contiguous()
        context=context.view(context.shape[0],context.shape[1],-1)
        return   context


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, args,mask_flag=True, scale=None,  output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.args=args
        self.factor = args.factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(args.dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern*(tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg
    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern *(tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_full(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)  #非快速版
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        return V.contiguous()

class FullAttention(nn.Module):
    def __init__(self,args):
        super(FullAttention, self).__init__()
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1. / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = torch.softmax(scale * scores, dim=-1)
        V = torch.einsum("bhls,bshd->blhd", A, values).contiguous()
        return V


class AttentionLayer(nn.Module):
    def __init__(self, args):
        super(AttentionLayer, self).__init__()
        d_model=args.d_model
        self.n_heads = args.n_heads
        if args.att_name == 'FullAttention':
            AttentionMode=FullAttention
        elif args.att_name == 'ProbSparseAttention':
            AttentionMode = ProbAttention
        elif args.att_name == 'AutoCorrelation':
            AttentionMode = AutoCorrelation
        else :
            raise Exception("选择合适的注意力机制名字")
        self.inner_correlation = AttentionMode(args)
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out= self.inner_correlation(
            queries,
            keys,
            values
        )
        out = out.view(B, L, -1)
        return self.out_projection(out)


class CrossAttentionLayer(nn.Module):
    def __init__(self, args):
        super(CrossAttentionLayer, self).__init__()
        d_model=args.d_model
        self.n_heads = args.n_heads
        if args.att_name == 'FullAttention':
            AttentionMode=FullAttention
        elif args.att_name == 'ProbSparseAttention':
            AttentionMode = ProbAttention
        elif args.att_name == 'AutoCorrelation':
            AttentionMode = AutoCorrelation
        else :
            raise Exception("选择合适的注意力机制名字")
        self.inner_correlation = AttentionMode(args)
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out= self.inner_correlation(
            queries,
            keys,
            values
        )
        out = out.view(B, L, -1)
        return self.out_projection(out)