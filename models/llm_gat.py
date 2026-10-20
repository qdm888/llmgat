import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BertTextEncoder(nn.Module):
    def __init__(self, pretrained='bert-base-uncased', finetune=True):
        super().__init__()
        self.model = BertModel.from_pretrained(pretrained)
        self.finetune = finetune

    def forward(self, text):
        # text: (batch, 3, seq_len) -> input_ids / attention_mask / token_type_ids
        input_ids = text[:, 0, :].long()
        input_mask = text[:, 1, :].float()
        segment_ids = text[:, 2, :].long()
        if self.finetune:
            return self.model(input_ids=input_ids, attention_mask=input_mask,
                              token_type_ids=segment_ids)[0]
        with torch.no_grad():
            return self.model(input_ids=input_ids, attention_mask=input_mask,
                              token_type_ids=segment_ids)[0]


class Conv1DProjection(nn.Module):
    """Project a modality sequence to L positions with d features."""

    def __init__(self, in_dim, out_dim, out_len, kernel_size=5):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.conv = nn.Conv1d(out_dim, out_dim, kernel_size,
                              padding=kernel_size // 2)
        self.pool = nn.AdaptiveAvgPool1d(out_len)
        self.act = nn.ELU()

    def forward(self, x):
        # x: (batch, n_m, d_m) -> (batch, L, d)
        x = self.proj(x)
        x = self.act(self.conv(x.transpose(1, 2)))
        return self.pool(x).transpose(1, 2)


def build_local_graph_mask(seq_len, window):
    """Boolean adjacency of the local multimodal graph.

    Nodes are ordered [text | vision | audio], each block of length L.
    An edge j -> i exists iff |i - j| <= window, for every ordered modality
    pair (intra- and inter-modal); i == j gives the self-connection.
    Rows are targets, columns are sources.
    """
    L = seq_len
    pos = torch.arange(L)
    band = (pos.view(-1, 1) - pos.view(1, -1)).abs() <= window
    mask = band.view(1, 1, L, L).expand(3, 3, L, L).reshape(3 * L, 3 * L)
    return mask.clone()


class GATLayer(nn.Module):
    """Multi-head graph attention over the local band graph.

    e_ij = LeakyReLU(a^T [W x_i || W x_j]), normalized over each target's
    neighborhood. Hidden layers concatenate the K head outputs; the last
    layer averages them.
    """

    def __init__(self, in_dim, out_dim, heads, dropout=0.3,
                 negative_slope=0.2, concat=True):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat
        self.dropout = dropout
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_tgt = nn.Parameter(torch.empty(heads, out_dim))
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_tgt)
        nn.init.xavier_uniform_(self.a_src)

    def forward(self, x, adj_mask, return_attn=False):
        # x: (batch, N, in_dim); adj_mask: (N, N) bool
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.heads, self.out_dim).permute(0, 2, 1, 3)

        score_tgt = (h * self.a_tgt.view(1, self.heads, 1, -1)).sum(-1)
        score_src = (h * self.a_src.view(1, self.heads, 1, -1)).sum(-1)
        e = self.leaky_relu(score_tgt.unsqueeze(-1) + score_src.unsqueeze(-2))

        e = e.masked_fill(~adj_mask.view(1, 1, N, N), torch.finfo(e.dtype).min)
        alpha = torch.softmax(e, dim=-1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.matmul(alpha, h)  # (B, K, N, out_dim)
        if self.concat:
            out = out.permute(0, 2, 1, 3).reshape(B, N, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)
        out = F.elu(out)
        if return_attn:
            return out, alpha
        return out


class ModalityAttentionReadout(nn.Module):
    """One learned query per modality; softmax over that modality's nodes."""

    def __init__(self, dim, num_modalities=3):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(num_modalities, dim))
        nn.init.xavier_uniform_(self.queries)

    def forward(self, x, seq_len):
        chunks = x.split(seq_len, dim=1)
        h_modal = []
        for m, xm in enumerate(chunks):
            scores = torch.einsum('bld,d->bl', xm, self.queries[m])
            alpha = torch.softmax(scores, dim=-1)
            h_modal.append(torch.einsum('bl,bld->bd', alpha, xm))
        return h_modal  # [h^t, h^v, h^a]


class LLM_GAT(nn.Module):
    """GAT-based dynamic fusion predictor (LLM-DA runs offline, see augmentation/)."""

    def __init__(self, args):
        super().__init__()
        self.use_bert = args.get('use_bert', True)
        if self.use_bert:
            self.bertmodel = BertTextEncoder(
                pretrained=args.get('pretrained', 'bert-base-uncased'),
                finetune=args.get('use_finetune', True))

        d_t, d_a, d_v = args['feature_dims']  # (text, audio, vision)
        self.d = d = args.get('proj_dim', 256)
        self.L = L = args.get('seq_len', 50)
        self.window = args.get('graph_window', 3)
        heads = args.get('gat_heads', 8)
        layers = args.get('gat_layers', 3)
        dropout = args.get('dropout', 0.3)
        head_dim = args.get('gat_head_dim', d // heads)

        k_t, k_a, k_v = args.get('conv_kernels', (5, 5, 5))
        self.proj_t = Conv1DProjection(d_t, d, L, kernel_size=k_t)
        self.proj_a = Conv1DProjection(d_a, d, L, kernel_size=k_a)
        self.proj_v = Conv1DProjection(d_v, d, L, kernel_size=k_v)

        self.fusion_mode = args.get('fusion_mode', 'gat')  # 'gat' or 'concat'
        if self.fusion_mode == 'gat':
            self.register_buffer('adj_mask',
                                 build_local_graph_mask(L, self.window))
            gat = []
            in_dim = d
            for layer in range(layers):
                last = layer == layers - 1
                gat.append(GATLayer(in_dim, head_dim, heads, dropout=dropout,
                                    concat=not last))
                in_dim = head_dim if last else heads * head_dim
            self.gat_layers = nn.ModuleList(gat)
            self.node_out = nn.Linear(in_dim, d) if in_dim != d else nn.Identity()
            self.readout = ModalityAttentionReadout(d)

        hidden = args.get('mlp_hidden', 256)
        self.classifier = nn.Sequential(
            nn.Linear(3 * d, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, audio, video, return_attn=False):
        h_t = self.bertmodel(text) if self.use_bert else text
        x_t = self.proj_t(h_t)  # (B, L, d)
        x_a = self.proj_a(audio)
        x_v = self.proj_v(video)

        attn_last = None
        if self.fusion_mode == 'gat':
            x = torch.cat([x_t, x_v, x_a], dim=1)  # 3L nodes ordered T | V | A
            x = self.dropout(x)
            for i, layer in enumerate(self.gat_layers):
                last = i == len(self.gat_layers) - 1
                if last and return_attn:
                    x, attn_last = layer(x, self.adj_mask, return_attn=True)
                else:
                    x = layer(x, self.adj_mask)
            x = self.node_out(x)
            h_t, h_v, h_a = self.readout(x, self.L)
        else:  # concat baseline: mean pooling per modality
            h_t = x_t.mean(dim=1)
            h_v = x_v.mean(dim=1)
            h_a = x_a.mean(dim=1)

        h_graph = torch.cat([h_t, h_v, h_a], dim=-1)
        output = self.classifier(h_graph)

        aux = {'h_t': h_t, 'h_v': h_v, 'h_a': h_a, 'h_graph': h_graph}
        if return_attn:
            aux['edge_attn'] = attn_last  # (B, K, 3L, 3L), last GAT layer
        return output, aux


def modality_consistency_loss(aux, eps=1e-8):
    """L_cons = 1 - ( cos(h^t, h^v) + cos(h^t, h^a) ) / 2"""
    cos_tv = F.cosine_similarity(aux['h_t'], aux['h_v'], dim=-1, eps=eps)
    cos_ta = F.cosine_similarity(aux['h_t'], aux['h_a'], dim=-1, eps=eps)
    return (1.0 - 0.5 * (cos_tv + cos_ta)).mean()
