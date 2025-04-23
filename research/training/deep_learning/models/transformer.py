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
from torch import nn


class MultiStepForecastTransformer(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 output_size: int,
                 target_horizon: int,
                 num_heads: int = 2,
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
