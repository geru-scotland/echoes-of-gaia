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
import json
import os
import traceback
from pathlib import Path

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import time
from tqdm.auto import tqdm
import numpy as np
import yaml
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

from research.training.deep_learning.models.lstm import BiomeLSTM, BiomeGRU, BiomeLSTMWithMultiheadAttention, \
    TransformerForecast
from research.training.deep_learning.data.dataset import SimulationDataset
from shared.enums.enums import NeuralMode
from shared.enums.strings import Loggers
from utils.loggers import LoggerManager
from utils.paths import BASE_DIR, DEEP_LEARNING_CONFIG_DIR, NEURAL_MODELS


class NeuralModelManager:
    def __init__(self, config_path: Optional[str] = None, mode: NeuralMode = NeuralMode.TRAINING):
        self._logger: logging.Logger = LoggerManager.get_logger(Loggers.DEEP_LEARNING)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._config = None
        self._data_path = None
        self._normalization_stats = {}

        base_config_path: str = os.path.join(DEEP_LEARNING_CONFIG_DIR, 'neural_config.yaml')
        config_path = config_path if config_path else base_config_path
        self._config = self._load_config(config_path)
        self._logger.info(f"Initialising Neural model manager with config: {config_path}")
        if mode == NeuralMode.INFERENCE:
            self.load_model()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self._logger.error(f"Error loading config: {e}")
            raise

    def _init_model(self) -> None:
        hyperparameters = self._config["hyperparameters"]
        self._model = TransformerForecast(
            input_size=len(self.config["data"]['features']),
            hidden_size=hyperparameters["hidden_size"],
            num_layers=hyperparameters["num_layers"],
            output_size=len(self.config['data']["targets"]),
            dropout=hyperparameters.get("dropout", 0.2),
            target_horizon=self.config["data"]["target_horizon"]
        ).to(self._device)

        self._logger.info(f"Initialized model: {self._model}")

    def load_model(self, model_path: Optional[str] = None) -> bool:
        if model_path is None:
            model_path = os.path.join(NEURAL_MODELS, self._config["paths"]["inference_model"])

        if not os.path.exists(model_path):
            self._logger.warning(f"Model file not found at {model_path}")
            return False
        self._logger.info(f"Loading Neural model: {model_path}")
        try:
            saved_dict = torch.load(model_path, map_location=self._device, weights_only=False)
            if self._model is None:
                config = saved_dict.get('config', {})
                target_horizon = config.get('target_horizon', 1)
                self._model = TransformerForecast(
                    input_size=config.get('input_size', config["input_size"]),
                    hidden_size=config.get('hidden_size', config["hidden_size"]),
                    num_layers=config.get('num_layers', config["num_layers"]),
                    output_size=config.get('output_size', config["output_size"]),
                    target_horizon=target_horizon
                ).to(self._device)

            self._model.load_state_dict(saved_dict['model_state_dict'])
            self._model.eval()

            self._normalization_stats = saved_dict.get('normalization_stats', None)

            self._logger.info(f"Model and parameters loaded from {model_path}")
            return True
        except Exception as e:
            tb = traceback.format_exc()
            self._logger.exception(f"Error loading model. Traceback: {tb}")
            self._logger.error(f"Error loading model: {e}")
            return False

    def save_model(self, model_path: Optional[str] = None) -> None:
        if model_path is None:
            model_path = os.path.join(BASE_DIR, self._config["paths"]["model_save_path"])

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        try:
            model_dict: Dict[str, Any] = {
                'model_state_dict': self._model.state_dict(),
                'normalization_stats': self._normalization_stats,
                'config': {
                    'input_size': len(self._config["data"]["features"]),
                    'output_size': len(self._config["data"]["targets"]),
                    'hidden_size': self._config["hyperparameters"]["hidden_size"],
                    'num_layers': self._config["hyperparameters"]["num_layers"],
                    'target_horizon': self._config["data"].get("target_horizon", 1)
                }
            }
            torch.save(model_dict, model_path)
            self._logger.info(f"Model, normalization stats and configs saved to {model_path}")
        except Exception as e:
            self._logger.error(f"Error saving model: {e}")

    def load_data(self, data_path: Optional[str] = None) -> List[Dict[str, Any]]:
        if data_path is None:
            data_path = os.path.join(BASE_DIR, self._config["paths"]["data_path"])

        self._logger.info(f"[LSTMModelManager] Loading data from {data_path}")
        data = []
        simulation_boundaries = []
        last_snapshot_id = -1

        try:
            for file_path in Path(data_path).glob('*.jsonl'):
                self._logger.info(f"Processing file: {file_path}")
                with open(file_path, 'r') as f:
                    for line_idx, line in enumerate(f):
                        if line.strip():
                            try:
                                entry = json.loads(line)
                                if 'data' in entry:
                                    snapshot_id = entry.get('snapshot_id', -1)

                                    if snapshot_id < last_snapshot_id:
                                        simulation_boundaries.append(len(data))

                                    last_snapshot_id = snapshot_id

                                    cleaned_data = {k: v for k, v in entry['data'].items()
                                                    if k not in ['snapshot_id', 'timestamp', 'simulation_time']}

                                    cleaned_data['_simulation_id'] = len(simulation_boundaries)
                                    data.append(cleaned_data)

                            except json.JSONDecodeError as e:
                                self._logger.warning(f"Invalid JSON in line {line_idx}: {e}")

                if data and (not simulation_boundaries or simulation_boundaries[-1] != len(data)):
                    simulation_boundaries.append(len(data))
                    last_snapshot_id = -1

            self._logger.info(f"Loaded {len(data)} data points from {len(simulation_boundaries)} simulations")
            return data, simulation_boundaries

        except Exception as e:
            self._logger.error(f"Error loading data: {e}")
            return [], []

    def train(self) -> Dict[str, List[float]]:
        if self._model is None:
            self._init_model()

        data, simulation_boundaries = self.load_data()

        if not data:
            self._logger.error("No data available for training")
            return {"train_loss": [], "val_loss": [], "train_mse": [], "val_mse": [], "train_r2": [], "val_r2": []}

        data_config = self._config["data"]
        hyperparameters = self._config["hyperparameters"]
        target_horizon = data_config.get("target_horizon", 1)

        train_dataset = SimulationDataset(
            data=data,
            sequence_length=data_config["sequence_length"],
            features=data_config["features"],
            targets=data_config["targets"],
            simulation_boundaries=simulation_boundaries,
            target_horizon=target_horizon
        )

        self._normalization_stats = train_dataset.get_normalization_stats()

        val_size = data_config["validation_split"] if data_config["validation_split"] > 0 else 0.2

        split_index = int(len(train_dataset) * (1 - val_size))

        self._logger.info(f"Total size of original dataset: {len(train_dataset)}")
        self._logger.info(f"Validation split ratio: {val_size}")
        self._logger.info(f"Split index: {split_index}")
        self._logger.info(f"Train subset: start=0, end={split_index - 1}, size={split_index}")
        self._logger.info(
            f"Validation subset: start={split_index}, end={len(train_dataset) - 1}, size={len(train_dataset) - split_index}")

        train_subset = torch.utils.data.Subset(train_dataset, list(range(0, split_index)))
        val_subset = torch.utils.data.Subset(train_dataset, list(range(split_index, len(train_dataset))))

        train_loader = DataLoader(
            train_subset,
            batch_size=hyperparameters["batch_size"],
            shuffle=False
        )

        val_loader = None
        if val_subset:
            val_loader = DataLoader(
                val_subset,
                batch_size=hyperparameters["batch_size"],
                shuffle=False
            )

        mse_criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            self._model.parameters(),
            lr=hyperparameters["learning_rate"],
            weight_decay=hyperparameters["weight_decay"]
        )

        total_steps = len(train_loader) * hyperparameters["num_epochs"]
        scheduler = OneCycleLR(
            optimizer,
            max_lr=hyperparameters["learning_rate"],
            total_steps=total_steps,
            pct_start=0.3,
            div_factor=25,  # LR inicial = max_lr/25
            final_div_factor=10000  # LR final = max_lr/10000
        )

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_mse": [],
            "val_mse": [],
            "train_r2": [],
            "val_r2": []
        }

        from sklearn.metrics import r2_score
        import time
        from tqdm.auto import tqdm

        use_tqdm = self._config.get("training", {}).get("use_progress_bar", True)

        for epoch in range(hyperparameters["num_epochs"]):
            start_time = time.time()

            self._model.train()
            train_loss = 0.0
            all_train_targets = []
            all_train_outputs = []

            if use_tqdm:
                train_iterator = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch + 1}/{hyperparameters['num_epochs']} [Train]",
                    leave=False,
                    ncols=100,
                    unit="batch"
                )
            else:
                train_iterator = train_loader
                self._logger.info(f"Epoch {epoch + 1}/{hyperparameters['num_epochs']} - Training...")

            batch_count = 0
            log_interval = max(1, len(train_loader) // 10)

            for inputs, targets in train_iterator:
                inputs = inputs.to(self._device)
                targets = targets.to(
                    self._device)

                outputs = self._model(inputs)

                loss = mse_criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

                all_train_targets.append(targets.cpu().detach().numpy())
                all_train_outputs.append(outputs.cpu().detach().numpy())

                if use_tqdm:
                    batch_count += 1
                    if batch_count % log_interval == 0 or batch_count == len(train_loader):
                        avg_loss_so_far = train_loss / batch_count
                        train_iterator.set_postfix(loss=f"{avg_loss_so_far:.4f}")

            all_train_targets = np.vstack(all_train_targets)
            all_train_outputs = np.vstack(all_train_outputs)

            if len(all_train_targets.shape) == 3:
                all_train_targets = all_train_targets.reshape(-1, all_train_targets.shape[-1])
                all_train_outputs = all_train_outputs.reshape(-1, all_train_outputs.shape[-1])

            train_mse = np.mean((all_train_outputs - all_train_targets) ** 2)
            train_r2 = r2_score(all_train_targets, all_train_outputs)
            avg_train_loss = train_loss / len(train_loader)

            history["train_loss"].append(avg_train_loss)
            history["train_mse"].append(train_mse)
            history["train_r2"].append(train_r2)

            if val_loader:
                self._model.eval()
                val_loss = 0.0
                all_val_targets = []
                all_val_outputs = []

                if use_tqdm:
                    val_iterator = tqdm(
                        val_loader,
                        desc=f"Epoch {epoch + 1}/{hyperparameters['num_epochs']} [Valid]",
                        leave=False,
                        ncols=100,
                        unit="batch"
                    )
                else:
                    val_iterator = val_loader
                    self._logger.info(f"Epoch {epoch + 1}/{hyperparameters['num_epochs']} - Validating...")

                batch_count = 0
                log_interval = max(1, len(val_loader) // 5)

                with torch.no_grad():
                    for inputs, targets in val_iterator:
                        inputs = inputs.to(self._device)
                        targets = targets.to(self._device)

                        outputs = self._model(inputs)
                        loss = mse_criterion(outputs, targets)
                        val_loss += loss.item()

                        all_val_targets.append(targets.cpu().numpy())
                        all_val_outputs.append(outputs.cpu().numpy())

                        if use_tqdm:
                            batch_count += 1
                            if batch_count % log_interval == 0 or batch_count == len(val_loader):
                                avg_loss_so_far = val_loss / batch_count
                                val_iterator.set_postfix(loss=f"{avg_loss_so_far:.4f}")

                all_val_targets = np.vstack(all_val_targets)
                all_val_outputs = np.vstack(all_val_outputs)

                if len(all_val_targets.shape) == 3:
                    all_val_targets = all_val_targets.reshape(-1, all_val_targets.shape[-1])
                    all_val_outputs = all_val_outputs.reshape(-1, all_val_outputs.shape[-1])

                val_mse = np.mean((all_val_outputs - all_val_targets) ** 2)
                val_r2 = r2_score(all_val_targets, all_val_outputs)

                avg_val_loss = val_loss / len(val_loader)

                history["val_loss"].append(avg_val_loss)
                history["val_mse"].append(val_mse)
                history["val_r2"].append(val_r2)

                epoch_time = time.time() - start_time
                batches = []
                for i, (inputs, targets) in enumerate(val_loader):
                    batches.append((inputs, targets))
                    if i >= 2:
                        break

                if batches:
                    batch_idx = random.randint(0, len(batches) - 1)
                    sample_inputs, sample_targets = batches[batch_idx]
                    sample_inputs = sample_inputs.to(self._device)
                else:
                    sample_inputs, sample_targets = next(iter(val_loader))
                    sample_inputs = sample_inputs.to(self._device)

                with torch.no_grad():
                    sample_outputs = self._model(sample_inputs)

                if epoch % 5 == 0:
                    sample_inputs, sample_targets = next(iter(val_loader))
                    sample_inputs = sample_inputs.to(self._device)
                    sample_targets = sample_targets.to(self._device)

                    with torch.no_grad():
                        sample_outputs = self._model(sample_inputs)

                    sample_targets_np = sample_targets.cpu().numpy()
                    sample_outputs_np = sample_outputs.cpu().numpy()

                    if self._normalization_stats and 'targets' in self._normalization_stats:
                        target_stats = self._normalization_stats['targets']

                        self._logger.info(f"Estadísticas de normalización: {target_stats}")

                        tmins = target_stats.get('mins', 0)
                        tmaxs = target_stats.get('maxs', 1)
                        trange = tmaxs - tmins

                        denorm_targets = np.zeros_like(sample_targets_np)
                        denorm_outputs = np.zeros_like(sample_outputs_np)

                        for step in range(sample_targets_np.shape[1]):
                            denorm_targets[:, step, :] = sample_targets_np[:, step, :] * trange + tmins
                            denorm_outputs[:, step, :] = sample_outputs_np[:, step, :] * trange + tmins
                    else:
                        self._logger.warning("Couldn't find any normalization stats")
                        denorm_targets = sample_targets_np
                        denorm_outputs = sample_outputs_np

                    example_idx = random.randrange(sample_targets.size(0))
                    self._logger.info(f"\n===== Ejemplo de predicción DESNORMALIZADA (Época {epoch + 1}) =====")

                    for step in range(denorm_targets.shape[1]):
                        step_error = np.abs(denorm_outputs[example_idx, step] - denorm_targets[example_idx, step])
                        step_mse = np.mean(step_error ** 2)

                        targets_str = ", ".join([f"{val:.2f}" for val in denorm_targets[example_idx, step]])
                        outputs_str = ", ".join([f"{val:.2f}" for val in denorm_outputs[example_idx, step]])
                        error_str = ", ".join([f"{val:.2f}" for val in step_error])

                        self._logger.info(f"Paso {step + 1}:")
                        self._logger.info(f"  Target real: [{targets_str}]")
                        self._logger.info(f"  Predicción: [{outputs_str}]")
                        self._logger.info(f"  Error Abs: [{error_str}]")
                        self._logger.info(f"  MSE: {step_mse:.4f}")

                    total_mse = np.mean((denorm_outputs[example_idx] - denorm_targets[example_idx]) ** 2)
                    self._logger.info(f"MSE Total para todos los pasos: {total_mse:.4f}")
                    self._logger.info("=" * 50)
                self._logger.info(
                    f"Epoch {epoch + 1}/{hyperparameters['num_epochs']} - "
                    f"Time: {epoch_time:.1f}s - "
                    f"Train Loss: {avg_train_loss:.4f} - "
                    f"Val Loss: {avg_val_loss:.4f} - "
                    f"Train MSE: {train_mse:.4f} - "
                    f"Val MSE: {val_mse:.4f} - "
                    f"Train R²: {train_r2:.4f} - "
                    f"Val R²: {val_r2:.4f}"
                )
            else:
                epoch_time = time.time() - start_time
                self._logger.info(
                    f"Epoch {epoch + 1}/{hyperparameters['num_epochs']} - "
                    f"Time: {epoch_time:.1f}s - "
                    f"Train Loss: {avg_train_loss:.4f} - "
                    f"Train MSE: {train_mse:.4f} - "
                    f"Train R²: {train_r2:.4f}"
                )

            current_lr = optimizer.param_groups[0]['lr']
            self._logger.info(f"Current LR: {current_lr:.6f}")

        self.save_model()

        if self._config.get("visualization", {}).get("enabled", True):
            plots_dir = os.path.join(BASE_DIR, self._config.get("paths", {}).get("plots_dir", "research/plots/lstm"))
            os.makedirs(plots_dir, exist_ok=True)

            import time
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            hyperparameters = self._config["hyperparameters"]
            plot_base_name = f"lstm_training_{timestamp}_h{hyperparameters['hidden_size']}_l{hyperparameters['num_layers']}"

            self._visualize_metrics(history, plot_base_name, plots_dir)
            self.analyze_correlation(data)

        return history

    def _normalize_sequence(self, sequence_array, feature_stats):
        mins = np.array(feature_stats['mins'])
        maxs = np.array(feature_stats['maxs'])
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0

        normalized = (sequence_array - mins) / ranges
        return np.clip(normalized, 0.0, 1.0)

    def predict(self, sequence: Union[List[Dict], np.ndarray]) -> np.ndarray:
        if self._model is None:
            loaded = self.load_model()
            if not loaded:
                raise ValueError("No model available for prediction")

        self._model.eval()

        train_config = self._config["data"]

        sequence_array = sequence

        if len(sequence_array) < train_config["sequence_length"]:
            raise ValueError(
                f"Input sequence length ({len(sequence_array)}) is shorter than required ({train_config['sequence_length']})")

        if len(sequence_array) > train_config["sequence_length"]:
            sequence_array = sequence_array[-train_config["sequence_length"]:]

        if self._normalization_stats.get("features"):
            sequence_array = self._normalize_sequence(
                sequence_array,
                self._normalization_stats["features"]
            )
        sequence_tensor = torch.tensor(sequence_array, dtype=torch.float32).unsqueeze(0).to(self._device)

        with torch.no_grad():
            prediction = self._model(sequence_tensor)

        #  [batch_size, target_horizon, output_size]
        prediction = prediction.cpu().numpy()

        if self._normalization_stats and 'targets' in self._normalization_stats:
            denormalized = self._denormalize_predictions(prediction, self._normalization_stats['targets'])
            return denormalized

        return prediction

    def _denormalize_predictions(self, predictions, target_stats):
        mins = target_stats.get('mins', 0)
        maxs = target_stats.get('maxs', 1)
        rng = maxs - mins

        original_shape = predictions.shape

        if len(original_shape) == 3:
            batch_size, horizon, num_features = original_shape
            reshaped = predictions.reshape(-1, num_features)
            denormalized = reshaped * rng + mins
            return denormalized.reshape(batch_size, horizon, num_features)

        else:
            return predictions * rng + mins

    def _visualize_metrics(self, history: Dict[str, List[float]], plot_base_name: str, plots_dir: str):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        sns.set_theme(style="whitegrid")

        metrics = [
            {"name": "Loss", "train": "train_loss", "val": "val_loss", "title": "Model Loss"},
            {"name": "MSE", "train": "train_mse", "val": "val_mse", "title": "Mean Squared Error"},
            {"name": "R²", "train": "train_r2", "val": "val_r2", "title": "R² Score"}
        ]

        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 5 * len(metrics)))

        if len(metrics) == 1:
            axes = [axes]

        hyperparameters = self._config["hyperparameters"]
        fig.suptitle(
            f"LSTM Model Training Metrics (h={hyperparameters['hidden_size']}, layers={hyperparameters['num_layers']})",
            fontsize=16)

        palette = sns.color_palette("viridis", 2)

        epochs = range(1, len(history['train_loss']) + 1)

        for i, metric in enumerate(metrics):
            ax = axes[i]

            train_values = history[metric["train"]]
            val_key = metric["val"]
            val_values = history.get(val_key, [])

            ax.plot(epochs, train_values, 'o-', color=palette[0],
                    label=f'Training {metric["name"]}', linewidth=2, markersize=5)

            if val_values:
                ax.plot(epochs, val_values, 'o-', color=palette[1],
                        label=f'Validation {metric["name"]}', linewidth=2, markersize=5)

            if len(epochs) > 3:
                try:
                    train_trend = np.polyfit(epochs, train_values, 3)
                    train_trend_fn = np.poly1d(train_trend)
                    x_smooth = np.linspace(min(epochs), max(epochs), 100)
                    ax.plot(x_smooth, train_trend_fn(x_smooth), '--', color=palette[0], alpha=0.5)

                    if val_values:
                        val_trend = np.polyfit(epochs, val_values, 3)
                        val_trend_fn = np.poly1d(val_trend)
                        ax.plot(x_smooth, val_trend_fn(x_smooth), '--', color=palette[1], alpha=0.5)
                except:
                    pass

            if metric["name"] == "R²":
                best_train = max(train_values)
                best_train_epoch = train_values.index(best_train) + 1
                ax.annotate(f'Max: {best_train:.4f}',
                            xy=(best_train_epoch, best_train),
                            xytext=(best_train_epoch, best_train * 0.9 if best_train > 0 else best_train * 1.1),
                            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                            fontsize=10, color=palette[0])

                if val_values:
                    best_val = max(val_values)
                    best_val_epoch = val_values.index(best_val) + 1
                    ax.annotate(f'Max: {best_val:.4f}',
                                xy=(best_val_epoch, best_val),
                                xytext=(best_val_epoch, best_val * 0.9 if best_val > 0 else best_val * 1.1),
                                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                                fontsize=10, color=palette[1])
            else:
                min_train = min(train_values)
                min_train_epoch = train_values.index(min_train) + 1
                ax.annotate(f'Min: {min_train:.4f}',
                            xy=(min_train_epoch, min_train),
                            xytext=(min_train_epoch, min_train * 1.1),
                            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                            fontsize=10, color=palette[0])

                if val_values:
                    min_val = min(val_values)
                    min_val_epoch = val_values.index(min_val) + 1
                    ax.annotate(f'Min: {min_val:.4f}',
                                xy=(min_val_epoch, min_val),
                                xytext=(min_val_epoch, min_val * 0.9),
                                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                                fontsize=10, color=palette[1])

            ax.set_title(metric["title"], fontsize=14)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric["name"], fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        all_metrics_path = os.path.join(plots_dir, f"{plot_base_name}_all_metrics.png")
        plt.savefig(all_metrics_path, dpi=300, bbox_inches='tight')
        self._logger.info(f"All metrics plot saved to: {all_metrics_path}")

        for i, metric in enumerate(metrics):
            plt.figure(figsize=(10, 6))
            train_values = history[metric["train"]]
            val_key = metric["val"]
            val_values = history.get(val_key, [])

            plt.plot(epochs, train_values, 'o-', color=palette[0],
                     label=f'Training {metric["name"]}', linewidth=2, markersize=5)

            if val_values:
                plt.plot(epochs, val_values, 'o-', color=palette[1],
                         label=f'Validation {metric["name"]}', linewidth=2, markersize=5)

            plt.title(metric["title"], fontsize=16)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel(metric["name"], fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            metric_path = os.path.join(plots_dir, f"{plot_base_name}_{metric['name'].lower()}.png")
            plt.savefig(metric_path, dpi=300, bbox_inches='tight')
            plt.close()

            self._logger.info(f"{metric['name']} plot saved to: {metric_path}")

        plt.close('all')

    def analyze_correlation(self, data: List[Dict[str, Any]]) -> None:
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        df = pd.DataFrame(data)

        cols_to_drop = ["_simulation_id"]
        for col in cols_to_drop:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        features = self._config["data"]["features"]
        targets = self._config["data"]["targets"]
        desired_columns = set(features + targets)

        existing_columns = [col for col in desired_columns if col in df.columns]
        df_selected = df[existing_columns]

        if df_selected.empty:
            self._logger.warning("No hay columnas numéricas seleccionadas para la correlación.")
            return

        correlation_matrix = df_selected.corr()

        self._logger.info("Matriz de Correlación:")
        self._logger.info("\n" + correlation_matrix.to_string())

        plt.figure(figsize=(10, 8))
        heatmap = sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            xticklabels=existing_columns,
            yticklabels=existing_columns
        )
        plt.xticks(rotation=45, ha="right")
        plt.title("Matriz de Correlación")
        plt.tight_layout()
        plt.show()

    @property
    def config(self):
        return self._config
