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

"""
Neural model manager for biome time-series forecasting and analysis.

Provides comprehensive infrastructure for training, evaluating, and deploying
LSTM and Transformer models for biome simulation predictions. Handles data 
loading, normalization, training pipelines, and visualization - supports both
training and inference modes with robust configuration management.
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
from sklearn.metrics import r2_score, explained_variance_score
import time
from tqdm.auto import tqdm
import numpy as np
import yaml
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

from research.training.deep_learning.models.lstm import BiomeLSTM
from research.training.deep_learning.models.transformer import MultiStepForecastTransformer
from research.training.deep_learning.data.dataset import SimulationDataset
from research.training.deep_learning.preprocess.ema import EMAProcessor
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

        # NOTA: Siempre mismo alpha que en training, que no se me olvide!
        self._ema_processor: EMAProcessor = EMAProcessor(alpha=0.05)

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
        self._model = BiomeLSTM(
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
                self._model = BiomeLSTM(
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

        total_samples = len(data)
        val_size = data_config["validation_split"] if data_config["validation_split"] > 0 else 0.2
        split_index = int(total_samples * (1 - val_size))

        train_data = data[:split_index]
        val_data = data[split_index:]

        self._logger.info(f"Total data length: {len(data)}")
        self._logger.info(f"Split index: {split_index}")
        self._logger.info(f"Train data length: {len(train_data)}")
        self._logger.info(f"Validation data length: {len(val_data)}")

        train_sim_boundaries = []
        val_sim_boundaries = []

        for boundary in simulation_boundaries:
            if boundary <= split_index:
                train_sim_boundaries.append(boundary)
                self._logger.info(f"Boundary {boundary} -> train (original: {boundary})")
            else:
                adjusted = boundary - split_index
                val_sim_boundaries.append(adjusted)
                self._logger.info(f"Boundary {boundary} -> validation (adjusted: {adjusted})")

        self._logger.info(f"Train simulation boundaries: {train_sim_boundaries}")
        self._logger.info(f"Validation simulation boundaries: {val_sim_boundaries}")

        train_dataset = SimulationDataset(
            data=train_data,
            sequence_length=data_config["sequence_length"],
            features=data_config["features"],
            targets=data_config["targets"],
            simulation_boundaries=train_sim_boundaries,
            target_horizon=target_horizon,
            normalization_method="minmax"
        )

        normalization_stats = train_dataset.get_normalization_stats()
        self._normalization_stats = normalization_stats

        # IMPORTANTE: Se me pasó al principio, val se tiene que normalizar
        # con los estadísticos (bueno, minmax al final uso) del train dataset.
        val_dataset = SimulationDataset(
            data=val_data,
            sequence_length=data_config["sequence_length"],
            features=data_config["features"],
            targets=data_config["targets"],
            simulation_boundaries=val_sim_boundaries,
            target_horizon=target_horizon,
            normalization_stats=normalization_stats
        )

        self._logger.info(f"=== DATA WINDOW CONFIGURATION ===")
        self._logger.info(f"Sequence length (input window): {data_config['sequence_length']}")
        self._logger.info(f"Target horizon (prediction steps): {target_horizon}")
        self._logger.info(f"Feature count: {len(data_config['features'])}")
        self._logger.info(f"Target count: {len(data_config['targets'])}")
        self._logger.info(f"Features: {data_config['features']}")
        self._logger.info(f"Targets: {data_config['targets']}")
        self._logger.info(f"Stride: 1")
        self._logger.info(f"Total dataset points: {total_samples}")
        self._logger.info(f"Total training points: {len(train_data)}")
        self._logger.info(f"Total validation points: {len(val_data)}")
        self._logger.info(f"Total training windows: {len(train_dataset)}")
        self._logger.info(f"Total validation windows: {len(val_dataset)}")
        self._logger.info(f"Training simulation boundaries: {train_sim_boundaries}")
        self._logger.info(f"Validation simulation boundaries: {val_sim_boundaries}")

        val_size = data_config["validation_split"] if data_config["validation_split"] > 0 else 0.2
        split_index = int(len(train_dataset) * (1 - val_size))

        self._logger.info(f"=== DATASET SPLIT ===")
        self._logger.info(f"Total dataset size: {len(train_dataset)}")
        self._logger.info(f"Validation split ratio: {val_size}")
        self._logger.info(f"Training subset: 0 to {split_index - 1} (size: {split_index})")
        self._logger.info(
            f"Validation subset: {split_index} to {len(train_dataset) - 1} (size: {len(train_dataset) - split_index})")

        train_loader = DataLoader(
            train_dataset,
            batch_size=hyperparameters["batch_size"],
            shuffle=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=hyperparameters["batch_size"],
            shuffle=False
        )

        baseline_metrics = self._precompute_baseline_metrics(val_loader, data_config, target_horizon)
        self._logger.info("=== BASELINE METRICS (CALCULATED ONCE) ===")
        self._logger.info(f"Baseline MSE: {baseline_metrics['mse']:.6f}")
        self._logger.info(f"Baseline MAE: {baseline_metrics['mae']:.6f}")
        self._logger.info(f"Baseline R²: {baseline_metrics['r2']:.6f}")

        huber_loss = nn.HuberLoss(delta=1.0, reduction='mean')
        mae_loss = nn.L1Loss(reduction='mean')
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
            pct_start=0.3,  # hasta el 30% sube, luego empieza decay ya
            div_factor=25,  # LR inicial = max_lr/25
            final_div_factor=10000  # LR final = max_lr/10000
        )

        history = {
            "train_loss": [], "val_loss": [],
            "train_mse": [], "val_mse": [],
            "train_rmse": [], "val_rmse": [],
            "train_r2": [], "val_r2": [],
            "train_exp_var": [], "val_exp_var": [],
            "train_mae": [], "val_mae": [],
            "baseline_mse": [], "baseline_mae": [], "baseline_r2": [], "baseline_rmse": []
        }

        history["baseline_mse"].append(baseline_metrics['mse'])
        history["baseline_rmse"].append(np.sqrt(baseline_metrics['mse']))
        history["baseline_mae"].append(baseline_metrics['mae'])
        history["baseline_r2"].append(baseline_metrics['r2'])

        from sklearn.metrics import r2_score, explained_variance_score
        import time
        from tqdm.auto import tqdm

        use_tqdm = self._config.get("training", {}).get("use_progress_bar", True)

        for epoch in range(hyperparameters["num_epochs"]):
            start_time = time.time()

            self._model.train()

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

            train_loss = 0.0
            all_train_targets = []
            all_train_outputs = []
            batch_count = 0
            log_interval = max(1, len(train_loader) // 10)

            for inputs, targets in train_iterator:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                outputs, _ = self._model(inputs)
                loss = 0.7 * huber_loss(outputs, targets) + 0.3 * mae_loss(outputs, targets)

                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

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

            avg_train_loss = train_loss / len(train_loader)
            train_mse = np.mean((all_train_outputs - all_train_targets) ** 2)
            train_mae = np.mean(np.abs(all_train_outputs - all_train_targets))
            train_r2 = r2_score(all_train_targets, all_train_outputs)
            train_rmse = np.sqrt(train_mse)
            train_exp_var = explained_variance_score(all_train_targets, all_train_outputs)

            history["train_loss"].append(avg_train_loss)
            history["train_mse"].append(train_mse)
            history["train_rmse"].append(train_rmse)
            history["train_exp_var"].append(train_exp_var)
            history["train_r2"].append(train_r2)
            history["train_mae"].append(train_mae)

            if val_loader:
                self._model.eval()
                all_val_targets = []
                all_val_outputs = []
                all_val_inputs = []  # Added for baseline

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
                val_loss = 0.0

                with torch.no_grad():
                    for inputs, targets in val_iterator:
                        inputs = inputs.to(self._device)
                        targets = targets.to(self._device)

                        all_val_inputs.append(inputs.cpu().numpy())

                        outputs, _ = self._model(inputs)
                        val_huber = huber_loss(outputs, targets)
                        val_mae = mae_loss(outputs, targets)
                        val_loss_total = 0.7 * val_huber + 0.3 * val_mae
                        val_loss += val_loss_total.item()

                        all_val_targets.append(targets.cpu().numpy())
                        all_val_outputs.append(outputs.cpu().numpy())

                        if use_tqdm:
                            batch_count += 1
                            if batch_count % log_interval == 0 or batch_count == len(val_loader):
                                avg_loss_so_far = val_loss / batch_count
                                val_iterator.set_postfix(loss=f"{avg_loss_so_far:.4f}")

                all_val_targets = np.vstack(all_val_targets)
                all_val_outputs = np.vstack(all_val_outputs)
                all_val_inputs = np.vstack(all_val_inputs)

                if len(all_val_targets.shape) == 3:
                    all_val_targets_flat = all_val_targets.reshape(-1, all_val_targets.shape[-1])
                    all_val_outputs_flat = all_val_outputs.reshape(-1, all_val_outputs.shape[-1])
                else:
                    all_val_targets_flat = all_val_targets
                    all_val_outputs_flat = all_val_outputs

                val_mse = np.mean((all_val_outputs_flat - all_val_targets_flat) ** 2)
                val_mae = np.mean(np.abs(all_val_outputs_flat - all_val_targets_flat))
                val_r2 = r2_score(all_val_targets_flat, all_val_outputs_flat)
                val_rmse = np.sqrt(val_mse)
                val_exp_var = explained_variance_score(all_val_targets_flat, all_val_outputs_flat)

                baseline_mse = baseline_metrics['mse']
                baseline_mae = baseline_metrics['mae']
                baseline_r2 = baseline_metrics['r2']

                if epoch > 0:
                    history["baseline_mse"].append(baseline_mse)
                    history["baseline_mae"].append(baseline_mae)
                    history["baseline_r2"].append(baseline_r2)

                mse_diff = baseline_mse - val_mse
                mae_diff = baseline_mae - val_mae
                r2_diff = val_r2 - baseline_r2

                history["val_loss"].append(val_loss / len(val_loader))
                history["val_mse"].append(val_mse)
                history["val_rmse"].append(val_rmse)
                history["val_exp_var"].append(val_exp_var)
                history["val_r2"].append(val_r2)
                history["val_mae"].append(val_mae)

                avg_val_loss = val_loss / len(val_loader)

                epoch_time = time.time() - start_time
                batches = []
                for i, (inputs, targets) in enumerate(val_loader):
                    batches.append((inputs, targets))
                    if i >= 2:
                        break

                if batches:
                    batch_idx = random.randint(0, len(batches) - 1)
                    sample_inputs, sample_targets = batches[batch_idx]
                else:
                    sample_inputs, sample_targets = next(iter(val_loader))
                    sample_inputs = sample_inputs.to(self._device)

                if epoch % 5 == 0 or epoch == hyperparameters["num_epochs"] - 1:
                    sample_inputs, sample_targets = next(iter(val_loader))
                    sample_inputs = sample_inputs.to(self._device)
                    sample_targets = sample_targets.to(self._device)

                    with torch.no_grad():
                        sample_outputs, _ = self._model(sample_inputs)

                    sample_targets_np = sample_targets.cpu().numpy()
                    sample_outputs_np = sample_outputs.cpu().numpy()

                    sample_inputs_np = sample_inputs.cpu().numpy()
                    sample_last_input = sample_inputs_np[:, -1, :]
                    sample_baseline = np.repeat(sample_last_input[:, None, :], sample_targets_np.shape[1], axis=1)
                    target_indices = [data_config["features"].index(t) for t in data_config["targets"] if
                                      t in data_config["features"]]
                    if target_indices:
                        sample_baseline = sample_baseline[:, :, target_indices]

                    if self._normalization_stats and 'targets' in self._normalization_stats:
                        target_stats = self._normalization_stats['targets']

                        self._logger.info(f"Normalization statistics: {target_stats}")

                        tmins = target_stats.get('mins', 0)
                        tmaxs = target_stats.get('maxs', 1)
                        trange = tmaxs - tmins

                        denorm_targets = np.zeros_like(sample_targets_np)
                        denorm_outputs = np.zeros_like(sample_outputs_np)
                        denorm_baseline = np.zeros_like(sample_baseline)

                        for step in range(sample_targets_np.shape[1]):
                            denorm_targets[:, step, :] = sample_targets_np[:, step, :] * trange + tmins
                            denorm_outputs[:, step, :] = sample_outputs_np[:, step, :] * trange + tmins
                            denorm_baseline[:, step, :] = sample_baseline[:, step, :] * trange + tmins
                    else:
                        self._logger.warning("Couldn't find any normalization stats")
                        denorm_targets = sample_targets_np
                        denorm_outputs = sample_outputs_np
                        denorm_baseline = sample_baseline

                    example_idx = random.randrange(sample_targets.size(0))
                    self._logger.info(f"\n===== DENORMALIZED Prediction Example (Epoch {epoch + 1}) =====")

                    model_total_mse = 0
                    baseline_total_mse = 0

                    for step in range(denorm_targets.shape[1]):
                        model_step_error = np.abs(denorm_outputs[example_idx, step] - denorm_targets[example_idx, step])
                        baseline_step_error = np.abs(
                            denorm_baseline[example_idx, step] - denorm_targets[example_idx, step])

                        model_step_mse = np.mean(model_step_error ** 2)
                        baseline_step_mse = np.mean(baseline_step_error ** 2)

                        model_total_mse += model_step_mse
                        baseline_total_mse += baseline_step_mse

                        targets_str = ", ".join([f"{val:.2f}" for val in denorm_targets[example_idx, step]])
                        outputs_str = ", ".join([f"{val:.2f}" for val in denorm_outputs[example_idx, step]])
                        baseline_str = ", ".join([f"{val:.2f}" for val in denorm_baseline[example_idx, step]])
                        model_error_str = ", ".join([f"{val:.2f}" for val in model_step_error])
                        baseline_error_str = ", ".join([f"{val:.2f}" for val in baseline_step_error])

                        self._logger.info(f"Step {step + 1}:")
                        self._logger.info(f"  Real target: [{targets_str}]")
                        self._logger.info(f"  Model prediction: [{outputs_str}]")
                        self._logger.info(f"  Baseline prediction: [{baseline_str}]")
                        self._logger.info(f"  Model Abs Error: [{model_error_str}]")
                        self._logger.info(f"  Baseline Abs Error: [{baseline_error_str}]")
                        self._logger.info(f"  Model MSE: {model_step_mse:.4f}")
                        self._logger.info(f"  Baseline MSE: {baseline_step_mse:.4f}")
                        model_step_mae = np.mean(np.abs(model_step_error))
                        baseline_step_mae = np.mean(np.abs(baseline_step_error))
                        self._logger.info(f"  Model MAE: {model_step_mae:.4f}")
                        self._logger.info(f"  Baseline MAE: {baseline_step_mae:.4f}")

                    model_total_mse /= denorm_targets.shape[1]
                    baseline_total_mse /= denorm_targets.shape[1]
                    self._logger.info(f"Model Total MSE for all steps: {model_total_mse:.4f}")
                    self._logger.info(f"Baseline Total MSE for all steps: {baseline_total_mse:.4f}")
                    improvement = ((baseline_total_mse - model_total_mse) / baseline_total_mse) * 100
                    self._logger.info(f"Improvement over baseline: {improvement:.2f}%")
                    self._logger.info("=" * 50)

                self._logger.info(
                    f"Epoch {epoch + 1}/{hyperparameters['num_epochs']} - "
                    f"Time: {epoch_time:.1f}s - "
                    f"Train Loss: {avg_train_loss:.4f} - "
                    f"Val Loss: {avg_val_loss:.4f} - "
                    f"Val RMSE: {val_rmse:.4f} - "
                    f"============ MODEL VS BASELINE COMPARISON ============"
                )
                self._logger.info(
                    f"MSE  => Model: {val_mse:.4f} | Baseline: {baseline_mse:.4f} | "
                    f"Difference: {mse_diff:.4f} | "
                    f"Improvement: {(mse_diff / baseline_mse * 100) if baseline_mse > 0 else 0:.2f}%"
                )
                self._logger.info(
                    f"RMSE => Model: {val_rmse:.4f} | Baseline: {np.sqrt(baseline_mse):.4f} | "
                    f"Difference: {(np.sqrt(baseline_mse) - val_rmse):.4f} | "
                    f"Improvement: {((np.sqrt(baseline_mse) - val_rmse) / np.sqrt(baseline_mse) * 100) if baseline_mse > 0 else 0:.2f}%"
                )
                self._logger.info(
                    f"MAE  => Model: {val_mae:.4f} | Baseline: {baseline_mae:.4f} | "
                    f"Difference: {mae_diff:.4f} | "
                    f"Improvement: {(mae_diff / baseline_mae * 100) if baseline_mae > 0 else 0:.2f}%"
                )
                self._logger.info(
                    f"R²   => Model: {val_r2:.4f} | Baseline: {baseline_r2:.4f} | "
                    f"Difference: {r2_diff:.4f}"
                )
                self._logger.info("=" * 70)

            else:
                epoch_time = time.time() - start_time
                self._logger.info(
                    f"Epoch {epoch + 1}/{hyperparameters['num_epochs']} - "
                    f"Time: {epoch_time:.1f}s - "
                    f"Train Loss: {avg_train_loss:.4f} - "
                    f"Train MSE: {train_mse:.4f} - "
                    f"Train MAE: {train_mae:.4f} - "
                    f"Train R²: {train_r2:.4f} - "
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

            if "baseline_mse" in history and history["baseline_mse"] and "val_mse" in history and history["val_mse"]:
                final_model_mse = history["val_mse"][-1]
                final_baseline_mse = history["baseline_mse"][-1]
                final_improvement = ((final_baseline_mse - final_model_mse) / final_baseline_mse) * 100

                self._logger.info("\n" + "=" * 30 + " FINAL SUMMARY " + "=" * 30)
                self._logger.info(f"Final model MSE: {final_model_mse:.6f}")
                self._logger.info(f"Final baseline MSE: {final_baseline_mse:.6f}")
                self._logger.info(f"Percentage improvement: {final_improvement:.2f}%")

                if final_improvement > 0:
                    self._logger.info("✅ MODEL OUTPERFORMS BASELINE")
                else:
                    self._logger.info("❌ MODEL DOES NOT OUTPERFORM BASELINE")
                self._logger.info("=" * 72)

            self._visualize_baseline_comparison(history, f"{plot_base_name}_vs_baseline", plots_dir)

            self._visualize_metrics(history, plot_base_name, plots_dir)

        return history

    def _precompute_baseline_metrics(self, val_loader, data_config, target_horizon):
        from sklearn.metrics import r2_score

        self._logger.info("Calculating baseline metrics...")
        all_val_targets = []
        all_val_inputs = []

        for inputs, targets in val_loader:
            all_val_targets.append(targets.cpu().numpy())
            all_val_inputs.append(inputs.cpu().numpy())

        all_val_targets = np.vstack(all_val_targets)
        all_val_inputs = np.vstack(all_val_inputs)

        if len(all_val_targets.shape) == 3:
            all_val_targets_flat = all_val_targets.reshape(-1, all_val_targets.shape[-1])

            all_val_inputs_last = all_val_inputs[:, -1, :]
            baseline_preds = np.repeat(all_val_inputs_last[:, None, :], all_val_targets.shape[1], axis=1)

            target_indices = [data_config["features"].index(t) for t in data_config["targets"]
                              if t in data_config["features"]]
            if target_indices:
                baseline_preds = baseline_preds[:, :, target_indices]

            baseline_preds_flat = baseline_preds.reshape(-1, baseline_preds.shape[-1])
        else:
            all_val_targets_flat = all_val_targets
            target_feature_idx = [data_config["features"].index(t) for t in data_config["targets"]]
            baseline_preds = all_val_inputs[:, -1, target_feature_idx]
            baseline_preds = np.repeat(baseline_preds, target_horizon, axis=1)
            baseline_preds_flat = baseline_preds

        baseline_mse = np.mean((baseline_preds_flat - all_val_targets_flat) ** 2)
        baseline_mae = np.mean(np.abs(baseline_preds_flat - all_val_targets_flat))

        if hasattr(self,
                   '_normalization_stats') and self._normalization_stats and 'targets' in self._normalization_stats:
            def denormalize(data, stats):
                return data * (stats['maxs'] - stats['mins']) + stats['mins']

            baseline_preds_denorm = denormalize(baseline_preds_flat, self._normalization_stats['targets'])
            targets_denorm = denormalize(all_val_targets_flat, self._normalization_stats['targets'])
            baseline_r2 = r2_score(targets_denorm, baseline_preds_denorm)
        else:
            baseline_r2 = r2_score(all_val_targets_flat, baseline_preds_flat)

        self._logger.info(f"Baseline MSE: {baseline_mse:.6f}")
        self._logger.info(f"Baseline MAE: {baseline_mae:.6f}")
        self._logger.info(f"Baseline R²: {baseline_r2:.6f}")

        return {
            'mse': baseline_mse,
            'mae': baseline_mae,
            'r2': baseline_r2,
            'predictions': baseline_preds_flat,
            'targets': all_val_targets_flat
        }

    def _visualize_baseline_comparison(self, history, base_name, plots_dir):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mtick
            import numpy as np
            from matplotlib.gridspec import GridSpec
            import matplotlib.colors as mcolors

            plt.style.use('seaborn-v0_8-whitegrid')

            colors = {
                'model': '#1f77b4',
                'baseline': '#d62728',
                'improvement': '#2ca02c',
                'model_rmse': '#ff7f0e',
                'model_mae': '#9467bd',
                'model_r2': '#8c564b',
                'model_filling': (0.12, 0.47, 0.71, 0.2),
                'baseline_filling': (0.84, 0.15, 0.16, 0.2),
                'grid': '#cccccc',
                'background': '#f9f9f9'
            }

            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
            plt.rcParams['font.size'] = 11
            plt.rcParams['axes.titlesize'] = 14
            plt.rcParams['axes.labelsize'] = 12

            has_mse = 'val_mse' in history and 'baseline_mse' in history and len(history['val_mse']) > 0
            has_rmse = 'val_rmse' in history and len(history['val_rmse']) > 0
            has_mae = 'val_mae' in history and 'baseline_mae' in history and len(history['val_mae']) > 0
            has_r2 = 'val_r2' in history and 'baseline_r2' in history and len(history['val_r2']) > 0

            if not (has_mse or has_rmse or has_mae or has_r2):
                self._logger.warning("No sufficient metrics data for visualization")
                return

            epochs = range(1, len(history['val_mse']) + 1) if has_mse else []

            fig = plt.figure(figsize=(14, 10), facecolor=colors['background'])
            gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.3)

            if has_mse:
                ax1 = fig.add_subplot(gs[0, 0])
                model_line, = ax1.plot(epochs, history['val_mse'],
                                       color=colors['model'], linewidth=2.5, marker='o',
                                       markersize=4, label='Model MSE')

                baseline_value = history['baseline_mse'][0]
                baseline_line = ax1.axhline(y=baseline_value, color=colors['baseline'],
                                            linestyle='--', linewidth=2, label='Baseline MSE')

                if all(m <= baseline_value for m in history['val_mse']):
                    ax1.fill_between(epochs, history['val_mse'], baseline_value,
                                     color=colors['model_filling'])

                final_mse = history['val_mse'][-1]
                ax1.annotate(f'Final: {final_mse:.4f}',
                             xy=(epochs[-1], final_mse),
                             xytext=(epochs[-1] - 2, final_mse * 1.1),
                             arrowprops=dict(arrowstyle='->', color='black', linewidth=1),
                             fontsize=9, fontweight='bold')

                ax1.annotate(f'Baseline: {baseline_value:.4f}',
                             xy=(epochs[-1], baseline_value),
                             xytext=(epochs[-1] - 2, baseline_value * 1.1),
                             arrowprops=dict(arrowstyle='->', color='black', linewidth=1),
                             fontsize=9, fontweight='bold')

                ax1.set_title('Mean Squared Error (MSE)', fontweight='bold')
                ax1.set_xlabel('Epochs')
                ax1.set_ylabel('MSE')
                ax1.legend(loc='upper right')
                ax1.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
                ax1.set_facecolor('white')

            if has_rmse:
                ax2 = fig.add_subplot(gs[0, 1])
                ax2.plot(epochs, history['val_rmse'], color=colors['model_rmse'],
                         linewidth=2.5, marker='o', markersize=4, label='Model RMSE')

                if 'baseline_mse' in history:
                    baseline_rmse = [np.sqrt(mse) for mse in history['baseline_mse']]
                    ax2.axhline(y=baseline_rmse[0], color=colors['baseline'],
                                linestyle='--', linewidth=2, label='Baseline RMSE')

                    if all(m <= baseline_rmse[0] for m in history['val_rmse']):
                        ax2.fill_between(epochs, history['val_rmse'], baseline_rmse[0],
                                         color=colors['model_filling'])

                    final_rmse = history['val_rmse'][-1]
                    ax2.annotate(f'Final: {final_rmse:.4f}',
                                 xy=(epochs[-1], final_rmse),
                                 xytext=(epochs[-1] - 2, final_rmse * 1.1),
                                 arrowprops=dict(arrowstyle='->', color='black', linewidth=1),
                                 fontsize=9, fontweight='bold')

                    ax2.annotate(f'Baseline: {baseline_rmse[0]:.4f}',
                                 xy=(epochs[-1], baseline_rmse[0]),
                                 xytext=(epochs[-1] - 2, baseline_rmse[0] * 1.1),
                                 arrowprops=dict(arrowstyle='->', color='black', linewidth=1),
                                 fontsize=9, fontweight='bold')

                ax2.set_title('Root Mean Squared Error (RMSE)', fontweight='bold')
                ax2.set_xlabel('Epochs')
                ax2.set_ylabel('RMSE')
                ax2.legend(loc='upper right')
                ax2.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
                ax2.set_facecolor('white')

            if has_mae:
                ax3 = fig.add_subplot(gs[1, 0])
                ax3.plot(epochs, history['val_mae'], color=colors['model_mae'],
                         linewidth=2.5, marker='o', markersize=4, label='Model MAE')

                baseline_mae = history['baseline_mae'][0]
                ax3.axhline(y=baseline_mae, color=colors['baseline'],
                            linestyle='--', linewidth=2, label='Baseline MAE')

                if all(m <= baseline_mae for m in history['val_mae']):
                    ax3.fill_between(epochs, history['val_mae'], baseline_mae,
                                     color=colors['model_filling'])

                final_mae = history['val_mae'][-1]
                ax3.annotate(f'Final: {final_mae:.4f}',
                             xy=(epochs[-1], final_mae),
                             xytext=(epochs[-1] - 2, final_mae * 1.1),
                             arrowprops=dict(arrowstyle='->', color='black', linewidth=1),
                             fontsize=9, fontweight='bold')

                ax3.annotate(f'Baseline: {baseline_mae:.4f}',
                             xy=(epochs[-1], baseline_mae),
                             xytext=(epochs[-1] - 2, baseline_mae * 1.1),
                             arrowprops=dict(arrowstyle='->', color='black', linewidth=1),
                             fontsize=9, fontweight='bold')

                ax3.set_title('Mean Absolute Error (MAE)', fontweight='bold')
                ax3.set_xlabel('Epochs')
                ax3.set_ylabel('MAE')
                ax3.legend(loc='upper right')
                ax3.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
                ax3.set_facecolor('white')

            if has_r2:
                ax4 = fig.add_subplot(gs[1, 1])
                ax4.plot(epochs, history['val_r2'], color=colors['model_r2'],
                         linewidth=2.5, marker='o', markersize=4, label='Model R²')

                baseline_r2 = history['baseline_r2'][0]
                ax4.axhline(y=baseline_r2, color=colors['baseline'],
                            linestyle='--', linewidth=2, label='Baseline R²')

                if all(r >= baseline_r2 for r in history['val_r2']):
                    ax4.fill_between(epochs, history['val_r2'], baseline_r2,
                                     color=colors['model_filling'])

                final_r2 = history['val_r2'][-1]
                ax4.annotate(f'Final: {final_r2:.4f}',
                             xy=(epochs[-1], final_r2),
                             xytext=(epochs[-1] - 2, final_r2 * 1.1),
                             arrowprops=dict(arrowstyle='->', color='black', linewidth=1),
                             fontsize=9, fontweight='bold')

                ax4.annotate(f'Baseline: {baseline_r2:.4f}',
                             xy=(epochs[-1], baseline_r2),
                             xytext=(epochs[-1] - 2, baseline_r2 * 1.1),
                             arrowprops=dict(arrowstyle='->', color='black', linewidth=1),
                             fontsize=9, fontweight='bold')

                ax4.set_title('Coefficient of Determination (R²)', fontweight='bold')
                ax4.set_xlabel('Epochs')
                ax4.set_ylabel('R²')
                ax4.legend(loc='lower right')
                ax4.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
                ax4.set_facecolor('white')

            fig.suptitle('Model Performance vs. Baseline', fontsize=16, fontweight='bold', y=0.98)

            hyperparameters = self._config["hyperparameters"]
            fig.text(0.5, 0.01,
                     f"Model: LSTM (hidden={hyperparameters.get('hidden_size', 'N/A')}, "
                     f"layers={hyperparameters.get('num_layers', 'N/A')}, "
                     f"lr={hyperparameters.get('learning_rate', 'N/A')})",
                     ha='center', fontsize=10)

            fig.savefig(os.path.join(plots_dir, f"{base_name}_metrics_comparison.png"),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)

            if has_mse:
                fig2 = plt.figure(figsize=(14, 8), facecolor=colors['background'])
                gs2 = GridSpec(2, 2, figure=fig2, wspace=0.25, hspace=0.3)

                baseline_mse = history['baseline_mse'][0]
                improvement = [(baseline_mse - mse) / baseline_mse * 100 if baseline_mse > 0 else 0
                               for mse in history['val_mse']]

                ax5 = fig2.add_subplot(gs2[0, 0])
                ax5.plot(epochs, improvement, color=colors['improvement'],
                         linewidth=2.5, marker='o', markersize=4)
                ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)

                ax5.fill_between(epochs, improvement, 0,
                                 where=[i > 0 for i in improvement],
                                 color=(0.12, 0.47, 0.71, 0.3),
                                 interpolate=True, label='Positive Improvement')

                ax5.fill_between(epochs, improvement, 0,
                                 where=[i <= 0 for i in improvement],
                                 color=(0.84, 0.15, 0.16, 0.3),
                                 interpolate=True, label='Negative Impact')

                final_improvement = improvement[-1]
                ax5.annotate(f'Final: {final_improvement:.2f}%',
                             xy=(epochs[-1], final_improvement),
                             xytext=(epochs[-1] - 3, final_improvement + 5),
                             arrowprops=dict(arrowstyle='->', color='black', linewidth=1),
                             fontsize=10, fontweight='bold')

                ax5.set_title('MSE Percentage Improvement Over Baseline', fontweight='bold')
                ax5.set_xlabel('Epochs')
                ax5.set_ylabel('Improvement (%)')
                ax5.yaxis.set_major_formatter(mtick.PercentFormatter())
                ax5.legend(loc='best')
                ax5.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
                ax5.set_facecolor('white')

                error_reduction = [baseline_mse - mse for mse in history['val_mse']]

                ax6 = fig2.add_subplot(gs2[0, 1])
                ax6.plot(epochs, error_reduction, color='#8c564b',
                         linewidth=2.5, marker='o', markersize=4)
                ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)

                ax6.fill_between(epochs, error_reduction, 0,
                                 where=[e > 0 for e in error_reduction],
                                 color=(0.12, 0.47, 0.71, 0.3),
                                 interpolate=True)

                ax6.fill_between(epochs, error_reduction, 0,
                                 where=[e <= 0 for e in error_reduction],
                                 color=(0.84, 0.15, 0.16, 0.3),
                                 interpolate=True)

                final_reduction = error_reduction[-1]
                ax6.annotate(f'Final: {final_reduction:.4f}',
                             xy=(epochs[-1], final_reduction),
                             xytext=(epochs[-1] - 3, final_reduction * 1.1),
                             arrowprops=dict(arrowstyle='->', color='black', linewidth=1),
                             fontsize=10, fontweight='bold')

                ax6.set_title('Absolute MSE Reduction vs Baseline', fontweight='bold')
                ax6.set_xlabel('Epochs')
                ax6.set_ylabel('MSE Reduction')
                ax6.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
                ax6.set_facecolor('white')

                relative_error = [mse / history['val_mse'][0] for mse in history['val_mse']]

                ax7 = fig2.add_subplot(gs2[1, 0])
                ax7.plot(epochs, relative_error, color='#17becf',
                         linewidth=2.5, marker='o', markersize=4)
                ax7.axhline(y=1, color='black', linestyle='-', linewidth=1)

                ax7.fill_between(epochs, relative_error, 1,
                                 where=[r < 1 for r in relative_error],
                                 color=(0.12, 0.47, 0.71, 0.3),
                                 interpolate=True)

                ax7.fill_between(epochs, relative_error, 1,
                                 where=[r >= 1 for r in relative_error],
                                 color=(0.84, 0.15, 0.16, 0.3),
                                 interpolate=True)

                final_relative = relative_error[-1]
                ax7.annotate(f'Final: {final_relative:.2f}x',
                             xy=(epochs[-1], final_relative),
                             xytext=(epochs[-1] - 3, final_relative * 1.1),
                             arrowprops=dict(arrowstyle='->', color='black', linewidth=1),
                             fontsize=10, fontweight='bold')

                ax7.set_title('Convergence Rate (Relative to Initial Error)', fontweight='bold')
                ax7.set_xlabel('Epochs')
                ax7.set_ylabel('Relative Error')
                ax7.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
                ax7.set_facecolor('white')

                if has_mae and has_r2:
                    mse_norm = [(baseline_mse - mse) / baseline_mse if baseline_mse > 0 else 0
                                for mse in history['val_mse']]

                    baseline_mae = history['baseline_mae'][0]
                    mae_norm = [(baseline_mae - mae) / baseline_mae if baseline_mae > 0 else 0
                                for mae in history['val_mae']]

                    baseline_r2 = history['baseline_r2'][0]
                    r2_norm = [(r2 - baseline_r2) / (1 - baseline_r2) if baseline_r2 < 1 else 0
                               for r2 in history['val_r2']]

                    combined_index = [0.4 * mse + 0.3 * mae + 0.3 * r2
                                      for mse, mae, r2 in zip(mse_norm, mae_norm, r2_norm)]

                    ax8 = fig2.add_subplot(gs2[1, 1])
                    ax8.plot(epochs, combined_index, color='#7f7f7f',
                             linewidth=2.5, marker='o', markersize=4)
                    ax8.axhline(y=0, color='black', linestyle='-', linewidth=1)

                    ax8.fill_between(epochs, combined_index, 0,
                                     where=[c > 0 for c in combined_index],
                                     color=(0.12, 0.47, 0.71, 0.3),
                                     interpolate=True)

                    ax8.fill_between(epochs, combined_index, 0,
                                     where=[c <= 0 for c in combined_index],
                                     color=(0.84, 0.15, 0.16, 0.3),
                                     interpolate=True)

                    final_combined = combined_index[-1]
                    ax8.annotate(f'Final: {final_combined:.4f}',
                                 xy=(epochs[-1], final_combined),
                                 xytext=(epochs[-1] - 3, final_combined * 1.1),
                                 arrowprops=dict(arrowstyle='->', color='black', linewidth=1),
                                 fontsize=10, fontweight='bold')

                    ax8.set_title('Combined Performance Index', fontweight='bold')
                    ax8.set_xlabel('Epochs')
                    ax8.set_ylabel('Performance Index')
                    ax8.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
                    ax8.set_facecolor('white')

                    ax8.text(0.5, -0.22,
                             "Combined Index = 0.4×MSE_improvement + 0.3×MAE_improvement + 0.3×R²_improvement",
                             transform=ax8.transAxes, ha='center',
                             fontsize=8, style='italic')

                fig2.suptitle('Model Improvement Analysis', fontsize=16, fontweight='bold', y=0.98)

                fig2.text(0.5, 0.01,
                          f"Baseline MSE: {baseline_mse:.6f} | "
                          f"Final Model MSE: {history['val_mse'][-1]:.6f} | "
                          f"Improvement: {final_improvement:.2f}%",
                          ha='center', fontsize=10)

                fig2.savefig(os.path.join(plots_dir, f"{base_name}_improvement_analysis.png"),
                             dpi=300, bbox_inches='tight')
                plt.close(fig2)

        except Exception as e:
            self._logger.error(f"Error visualizing baseline comparison: {e}")
            import traceback
            self._logger.error(traceback.format_exc())

    def _normalize_sequence(self, sequence_array, feature_stats, method='minmax'):
        if method == 'minmax':
            mins = np.array(feature_stats['mins'])
            maxs = np.array(feature_stats['maxs'])
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0

            normalized = (sequence_array - mins) / ranges
            return np.clip(normalized, 0.0, 1.0)

        elif method == 'zscore':
            means = np.array(feature_stats['means'])
            stds = np.array(feature_stats['stds'])
            stds[stds == 0] = 1.0

            normalized = (sequence_array - means) / stds
            return normalized

        else:
            raise ValueError(f"Unknown normalization method: {method}. Use 'minmax' or 'standard'.")

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

        sequence_array = self._ema_processor.process_sequence(sequence_array)

        if self._normalization_stats.get("features"):
            sequence_array = self._normalize_sequence(
                sequence_array,
                self._normalization_stats["features"],
                method="minmax"
            )
        sequence_tensor = torch.tensor(sequence_array, dtype=torch.float32).unsqueeze(0).to(self._device)

        with torch.no_grad():
            prediction, _ = self._model(sequence_tensor)

        #  [batch_size, target_horizon, output_size]
        prediction = prediction.cpu().numpy()

        if self._normalization_stats and 'targets' in self._normalization_stats:
            denormalized = self._denormalize_predictions(prediction, self._normalization_stats['targets'],
                                                         method="minmax")
            return denormalized

        return prediction

    def _denormalize_predictions(self, predictions, target_stats, method='minmax'):
        original_shape = predictions.shape

        if method == 'minmax':
            mins = target_stats.get('mins', 0)
            maxs = target_stats.get('maxs', 1)
            rng = maxs - mins

            if len(original_shape) == 3:
                batch_size, horizon, num_features = original_shape
                reshaped = predictions.reshape(-1, num_features)
                denormalized = reshaped * rng + mins
                return denormalized.reshape(batch_size, horizon, num_features)
            else:
                return predictions * rng + mins

        elif method == 'zscore':
            means = target_stats.get('means', 0)
            stds = target_stats.get('stds', 1)

            if len(original_shape) == 3:
                batch_size, horizon, num_features = original_shape
                reshaped = predictions.reshape(-1, num_features)
                denormalized = reshaped * stds + means
                return denormalized.reshape(batch_size, horizon, num_features)
            else:
                return predictions * stds + means

        else:
            raise ValueError(f"Unknown normalization method: {method}. Use 'minmax' or 'standard'.")

    def _visualize_metrics(self, history: Dict[str, List[float]], plot_base_name: str, plots_dir: str):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        sns.set_theme(style="whitegrid")

        metrics = [
            {"name": "Loss", "train": "train_loss", "val": "val_loss", "title": "Model Loss"},
            {"name": "MSE", "train": "train_mse", "val": "val_mse", "title": "Mean Squared Error"},
            {"name": "MAE", "train": "train_mae", "val": "val_mae", "title": "Mean Absolute Error"},
            {"name": "R²", "train": "train_r2", "val": "val_r2", "title": "R² Score"}
        ]

        available_metrics = []
        for metric in metrics:
            if metric["train"] in history and len(history[metric["train"]]) > 0:
                available_metrics.append(metric)

        if not available_metrics:
            self._logger.warning("No metrics available to visualize")
            return

        fig, axes = plt.subplots(len(available_metrics), 1, figsize=(12, 5 * len(available_metrics)))

        if len(available_metrics) == 1:
            axes = [axes]

        hyperparameters = self._config["hyperparameters"]
        fig.suptitle(
            f"LSTM Model Training Metrics (h={hyperparameters['hidden_size']}, layers={hyperparameters['num_layers']})",
            fontsize=16)

        palette = sns.color_palette("viridis", 2)

        for i, metric in enumerate(available_metrics):
            train_key = metric["train"]
            train_values = history.get(train_key, [])

            if not train_values:
                self._logger.warning(f"No data for {train_key}, skipping plot")
                continue

            epochs = range(1, len(train_values) + 1)
            ax = axes[i]

            ax.plot(epochs, train_values, 'o-', color=palette[0],
                    label=f'Training {metric["name"]}', linewidth=2, markersize=5)

            val_key = metric["val"]
            val_values = history.get(val_key, [])

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
                except Exception as e:
                    self._logger.warning(f"Could not fit trend line: {e}")

            if metric["name"] == "R²" and train_values:
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
            elif train_values:
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

        for metric in available_metrics:
            train_key = metric["train"]
            train_values = history.get(train_key, [])

            if not train_values:
                continue

            epochs = range(1, len(train_values) + 1)

            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_values, 'o-', color=palette[0],
                     label=f'Training {metric["name"]}', linewidth=2, markersize=5)

            val_key = metric["val"]
            val_values = history.get(val_key, [])

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
