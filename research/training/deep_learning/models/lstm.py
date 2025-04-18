""" 
# =============================================================================
#                                                                              #
#                              ✦ ECHOES OF GAIA ✦                              #
#                                                                              #
#    Trabajo Fin de Grado (TFG)                                                #
#    Facultad de Ingeniería Informática - Donostia                             #
#    UPV/EHU - Euskal Herriko Unibertsitatea                                   #
#                                                                              #
#    Área de Computación e Inteligencia Artificial                             #
#                                                                              #
#    Autor:  Aingeru García Blas                                               #
#    GitHub: https://github.com/geru-scotland                                  #
#    Repo:   https://github.com/geru-scotland/echoes-of-gaia                   #
#                                                                              #
# =============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List


class BiomeLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2,
                 target_horizon: int = 1,
                 num_heads: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.target_horizon = target_horizon
        self.output_size = output_size
        # TODO: Inicialización de pesos, etc.
        # Acordarte de batch/layer norm.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.fc = nn.Linear(hidden_size, output_size * target_horizon)
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)

        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            hidden = (h0, c0)

        lstm_out, hidden = self.lstm(x, hidden)
        # lstm_out shape: (batch_size, sequence_length, hidden_size)

        # Aplico atención, ojo, query, key, value son iguales
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        # attn_output shape: (batch_size, sequence_length, hidden_size)

        out = self.layer_norm(attn_output)

        # Hago pooling tomo la media sobre la dimensión temporal de seq
        out = out.mean(dim=1)

        out = self.fc(out)

        return out, hidden



class TransformerForecast(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 output_size: int,
                 target_horizon: int,
                 num_heads: int = 6,
                 dropout: float = 0.2,
                 max_seq_len: int = 30,
                 pool_k: int = 3
                 ):
        super().__init__()

        self.hidden_size = hidden_size
        self.target_horizon = target_horizon
        self.pool_k = pool_k

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.positional_embeddings = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_size)
        )
        self.step_embeddings = nn.Embedding(target_horizon, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_layers)

        self.output_layer = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_normalization = nn.LayerNorm(hidden_size)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_sequence):
        # Voy a poner bien las shapes para todo, quitar antes de defensa.
        # que me hecho la picha un lio.

        # input_sequence: (batch_size, seq_length, input_size)
        batch_size, seq_length, _ = input_sequence.size()

        # 1) Embedding + Positional Encoding

        # (batch_size, seq_length, hidden_size)
        embedded_sequence = self.input_layer(input_sequence)
        # (batch_size, seq_length, hidden_size)
        positional_encoded = embedded_sequence + self.positional_embeddings[:, :seq_length, :]

        # (batch_size, seq_length, hidden_size)
        encoder_output = self.encoder(positional_encoded)

        # 2) Mean‑pooling en últimas K posiciones

        k_steps = min(self.pool_k, seq_length)
        # (batch_size, k_steps, hidden_size)
        last_k_outputs = encoder_output[:, -k_steps:, :]
        # (batch_size, hidden_size)
        pooled_output = last_k_outputs.mean(dim=1)
        # (batch_size, hidden_size)
        normalized_pooled = self.layer_normalization(pooled_output)

        # 3) Embeddings para paso futuro, horizon pasos, cada uno con sus targets pongo.
        # nuevo tensor [0, 1, ..., target_horizon-1]
        step_indices = torch.arange(self.target_horizon, device=encoder_output.device)
        # (target_horizon, hidden_size)
        future_step_embeddings = self.step_embeddings(step_indices)

        # Al embedding aprendido para cada futuro paso le sumo el contexto norm. y pooled del encoder.
        # (batch_size, target_horizon, hidden_size)
        context_per_step = normalized_pooled.unsqueeze(1) + future_step_embeddings.unsqueeze(0)

        # 4) Proyecto ya a la capa de salida
        # Aplano para: (batch_size * target_horizon, hidden_size)
        flat_context = context_per_step.reshape(-1, self.hidden_size)
        # (batch_size * target_horizon, output_size)
        flat_projection = self.output_layer(flat_context)

        # (batch_size, target_horizon, output_size)
        forecasts = flat_projection.view(batch_size, self.target_horizon, -1)

        return forecasts

# Pongo GRU aquí mismo, por ahora. Cuando esté funcional y permita decidir, cambio a fichero separado.
class BiomeGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)

        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, hidden = self.gru(x, hidden)
        # out shape: (batch_size, sequence_length, hidden_size)

        out = self.layer_norm(out[:, -1, :])

        out = self.fc(out)
        # out shape: (batch_size, output_size)

        return out, hidden
