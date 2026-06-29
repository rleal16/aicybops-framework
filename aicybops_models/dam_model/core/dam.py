import torch
import torch.nn as nn
import torch.nn.functional as F

class DAMModel(nn.Module):
    def __init__(
        self,
        load_metrics_dim,
        traffic_metrics_dim,
        log_seq_dim,
        lstm_hidden_dim=32,
        num_heads=4,
        attn_dropout=0.0,
    ):
        super().__init__()
        self.load_metrics_dim = load_metrics_dim
        self.traffic_metrics_dim = traffic_metrics_dim
        self.log_seq_dim = log_seq_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        
        # LSTM branches.
        self.lstm_load_metrics = nn.LSTM(input_size=self.load_metrics_dim, hidden_size=self.lstm_hidden_dim, batch_first=True)
        self.lstm_traffic_metrics = nn.LSTM(input_size=self.traffic_metrics_dim, hidden_size=self.lstm_hidden_dim, batch_first=True)
        self.lstm_log_seq = nn.LSTM(input_size=self.log_seq_dim, hidden_size=self.lstm_hidden_dim, batch_first=True)
        self.attention_dim = 3 * self.lstm_hidden_dim
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {self.num_heads}")
        if self.attention_dim % self.num_heads != 0:
            raise ValueError(
                f"attention_dim ({self.attention_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        self.multihead_q_proj = nn.Linear(self.attention_dim, self.attention_dim)
        self.multihead_k_proj = nn.Linear(self.attention_dim, self.attention_dim)
        self.multihead_v_proj = nn.Linear(self.attention_dim, self.attention_dim)
        self.multihead_out_proj = nn.Linear(self.attention_dim, self.attention_dim)

        # Output heads.
        self.fc_load = nn.Sequential(
            nn.Linear(3 * self.lstm_hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, self.load_metrics_dim)
        )
        self.fc_traffic = nn.Sequential(
            nn.Linear(3 * self.lstm_hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, self.traffic_metrics_dim)
        )
        self.fc_log = nn.Sequential(
            nn.Linear(3 * self.lstm_hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, self.log_seq_dim)
        )

    def get_dimensions(self):
        """Return model dimensions."""
        return {
            "load_metrics_dim": self.load_metrics_dim,
            "traffic_metrics_dim": self.traffic_metrics_dim,
            "log_seq_dim": self.log_seq_dim,
            "lstm_hidden_dim": self.lstm_hidden_dim,
            "num_heads": self.num_heads,
            "attn_dropout": self.attn_dropout,
        }

    @property
    def dimensions(self):
        return self.get_dimensions()

    def _get_attention(self, query, key, value):
        """Single-head attention."""
        scale = query.size(-1) ** 0.5
        attention_scores = torch.bmm(query, key.transpose(1, 2)) 
        attention_weights = F.softmax(attention_scores / scale, dim=-1)
        attended = torch.bmm(attention_weights, value)
        return attended, attention_weights

    def _get_multihead_attention(self, query, key, value):
        """Multi-head scaled dot-product attention."""
        batch_size, query_len, embed_dim = query.size()
        _, key_len, key_dim = key.size()
        _, value_len, value_dim = value.size()

        if key_len != value_len:
            raise ValueError(
                f"key/value sequence mismatch: key_len={key_len}, value_len={value_len}"
            )
        if embed_dim != key_dim or embed_dim != value_dim:
            raise ValueError(
                "query/key/value must share the same embedding dimension, "
                f"got query={embed_dim}, key={key_dim}, value={value_dim}"
            )
        if embed_dim != self.attention_dim:
            raise ValueError(
                f"Expected attention embedding dim {self.attention_dim}, got {embed_dim}"
            )

        head_dim = embed_dim // self.num_heads
        scale = head_dim ** 0.5

        query = self.multihead_q_proj(query)
        key = self.multihead_k_proj(key)
        value = self.multihead_v_proj(value)

        # Reshape to [batch, heads, seq, head_dim] for per-head attention.
        q = query.view(batch_size, query_len, self.num_heads, head_dim).transpose(1, 2)
        k = key.view(batch_size, key_len, self.num_heads, head_dim).transpose(1, 2)
        v = value.view(batch_size, value_len, self.num_heads, head_dim).transpose(1, 2)

        # Flatten heads into batch dimension for efficient batched matmul.
        q = q.contiguous().view(batch_size * self.num_heads, query_len, head_dim)
        k = k.contiguous().view(batch_size * self.num_heads, key_len, head_dim)
        v = v.contiguous().view(batch_size * self.num_heads, value_len, head_dim)

        attention_scores = torch.bmm(q, k.transpose(1, 2))
        attention_weights = F.softmax(attention_scores / scale, dim=-1)
        attention_weights = F.dropout(attention_weights, p=self.attn_dropout, training=self.training)
        attended = torch.bmm(attention_weights, v)

        attended = attended.view(batch_size, self.num_heads, query_len, head_dim)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, query_len, embed_dim)
        attended = self.multihead_out_proj(attended)
        attention_weights = attention_weights.view(batch_size, self.num_heads, query_len, key_len)

        return attended, attention_weights

    def forward(self, load_seq, traffic_seq, log_seq, multihead=False):
        # LSTM branches.
        out_load, (h_load, _) = self.lstm_load_metrics(load_seq)
        out_traffic, (h_traffic, _) = self.lstm_traffic_metrics(traffic_seq)
        out_log, (h_log, _) = self.lstm_log_seq(log_seq)

        # Concatenate final hidden states.
        h_concat = torch.cat((h_load, h_traffic, h_log), dim=2)
        h_concat = h_concat.permute(1, 0, 2)

        # Concatenate LSTM outputs.
        out_concat = torch.cat((out_load, out_traffic, out_log), dim=2)

        if multihead:
            attended, _ = self._get_multihead_attention(
                h_concat,
                out_concat,
                out_concat,
            )
        else:
            attended, _ = self._get_attention(h_concat, out_concat, out_concat)
        attended = attended.squeeze(1)

        # Run dense heads.
        pred_load = self.fc_load(attended)
        pred_traffic = self.fc_traffic(attended)
        pred_log = self.fc_log(attended)

        return pred_load, pred_traffic, pred_log

    def predict(
        self,
        load_seq: torch.Tensor,
        traffic_seq: torch.Tensor,
        log_seq: torch.Tensor,
        device: torch.device = None,
        use_inference_mode: bool = True,
        multihead: bool = False
    ) -> tuple:
        """
        Run model inference for the given input sequences.

        Args:
            load_seq: Tensor of shape (batch, window, load_metrics_dim)
            traffic_seq: Tensor of shape (batch, window, traffic_metrics_dim)
            log_seq: Tensor of shape (batch, window, log_seq_dim)
            device: torch.device, if provided moves model and data to device
            use_inference_mode: if True, uses torch.inference_mode() for efficiency
            multihead: if True, uses multi-head attention; if False, single-head attention

        Returns:
            Tuple of predicted tensors: (pred_load, pred_traffic, pred_log)
        """
        self.eval()
        if device is not None:
            self.to(device)
            load_seq = load_seq.to(device)
            traffic_seq = traffic_seq.to(device)
            log_seq = log_seq.to(device)
        if use_inference_mode:
            with torch.inference_mode():
                return self.forward(
                    load_seq,
                    traffic_seq,
                    log_seq,
                    multihead=multihead,
                )
        else:
            with torch.no_grad():
                return self.forward(
                    load_seq,
                    traffic_seq,
                    log_seq,
                    multihead=multihead,
                )

