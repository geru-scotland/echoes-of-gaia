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
        self._model = MultiStepForecastTransformer(
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
                self._model = MultiStepForecastTransformer(
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
            normalization_method="minmax",
            use_deltas=True
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
            normalization_stats=normalization_stats,
            use_deltas=True
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

        # Me quedo con Huber al final, probé también SmoothL1.
        # a parte de combinar MSE + MAE ponderadas etc.
        huber_loss = nn.HuberLoss(delta=1.0, reduction='mean')

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
            "train_loss": [],
            "val_loss": [],
            "train_mse": [],
            "val_mse": [],
            "train_r2": [],
            "val_r2": [],
            "train_mae": [],
            "val_mae": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "train_exp_var": [], "val_exp_var": [],
            "train_nrmse": [], "val_nrmse": []
        }
        target_stats = normalization_stats.get("targets", {})
        tmins = np.array(target_stats.get("mins", 0))
        tmaxs = np.array(target_stats.get("maxs", 1))
        trange = tmaxs - tmins

        from sklearn.metrics import r2_score
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
                targets = targets.to(
                    self._device)

                outputs = self._model(inputs)

                loss = huber_loss(outputs, targets)

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
            if train_dataset._use_deltas:
                delta_train_mse = np.mean((all_train_outputs - all_train_targets) ** 2)
                delta_train_mae = np.mean(np.abs(all_train_outputs - all_train_targets))

                denorm_train_targets = all_train_targets * trange + tmins
                denorm_train_outputs = all_train_outputs * trange + tmins

                T = denorm_train_targets.reshape(-1, denorm_train_targets.shape[-1])
                P = denorm_train_outputs.reshape(-1, denorm_train_outputs.shape[-1])
                train_r2_per_dim = [r2_score(T[:, i], P[:, i]) for i in range(T.shape[1])]
                train_r2 = float(np.mean(train_r2_per_dim))
                self._logger.info(f"Train R² por dimensión: {train_r2_per_dim}")

                train_exp_var = explained_variance_score(denorm_train_targets, denorm_train_outputs)
                train_nrmse = np.sqrt(delta_train_mse) / (np.std(denorm_train_targets) + 1e-8)

                history["train_exp_var"].append(train_exp_var)
                history["train_nrmse"].append(train_nrmse)

                delta_train_accuracy = self._calculate_accuracy(all_train_outputs, all_train_targets)

                self._logger.info(
                    f"Train Delta Metrics - MSE: {delta_train_mse:.4f}, MAE: {delta_train_mae:.4f}, R²: {train_r2:.4f}")

                train_mse = delta_train_mse
                train_mae = delta_train_mae
                train_accuracy = delta_train_accuracy

            else:
                train_mse = np.mean((all_train_outputs - all_train_targets) ** 2)
                train_mae = np.mean(np.abs(all_train_outputs - all_train_targets))
                train_r2 = r2_score(all_train_targets, all_train_outputs)
                train_accuracy = self._calculate_accuracy(all_train_outputs, all_train_targets)

            history["train_loss"].append(avg_train_loss)
            history["train_mse"].append(train_mse)
            history["train_mae"].append(train_mae)
            history["train_r2"].append(train_r2)
            history["train_accuracy"].append(train_accuracy)

            if val_loader:
                self._model.eval()
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
                val_loss = 0.0

                with torch.no_grad():
                    for inputs, targets in val_iterator:
                        inputs = inputs.to(self._device)
                        targets = targets.to(self._device)

                        outputs = self._model(inputs)
                        hubber_val = huber_loss(outputs, targets)

                        val_loss += hubber_val.item()

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

                if val_dataset._use_deltas:
                    delta_val_mse = np.mean((all_val_outputs - all_val_targets) ** 2)
                    delta_val_mae = np.mean(np.abs(all_val_outputs - all_val_targets))

                    denorm_val_targets = all_val_targets * trange + tmins
                    denorm_val_outputs = all_val_outputs * trange + tmins

                    val_r2_per_dim = [
                        r2_score(denorm_val_targets[:, i], denorm_val_outputs[:, i])
                        for i in range(denorm_val_targets.shape[1])
                    ]
                    val_r2 = float(np.mean(val_r2_per_dim))
                    val_exp_var = explained_variance_score(denorm_val_targets, denorm_val_outputs)
                    val_nrmse = np.sqrt(delta_val_mse) / (np.max(denorm_val_targets) - np.min(denorm_val_targets))

                    history["val_exp_var"].append(val_exp_var)
                    history["val_nrmse"].append(val_nrmse)

                    delta_val_accuracy = self._calculate_accuracy(all_val_outputs, all_val_targets)

                    self._logger.info(
                        f"Val Delta Metrics - MSE: {delta_val_mse:.4f}, MAE: {delta_val_mae:.4f}, R²: {val_r2:.4f}")

                    val_mse = delta_val_mse
                    val_mae = delta_val_mae
                    val_accuracy = delta_val_accuracy
                else:
                    val_mse = np.mean((all_val_outputs - all_val_targets) ** 2)
                    val_mae = np.mean(np.abs(all_val_outputs - all_val_targets))
                    val_r2 = r2_score(all_val_targets, all_val_outputs)
                    val_accuracy = self._calculate_accuracy(all_val_outputs, all_val_targets)

                avg_val_loss = val_loss / len(val_loader)

                history["val_loss"].append(avg_val_loss)
                history["val_mse"].append(val_mse)
                history["val_mae"].append(val_mae)
                history["val_r2"].append(val_r2)
                history["val_accuracy"].append(val_accuracy)

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

                if epoch % 5 == 0:
                    sample_inputs, sample_targets = next(iter(val_loader))
                    sample_inputs = sample_inputs.to(self._device)
                    sample_targets = sample_targets.to(self._device)

                    with torch.no_grad():
                        sample_outputs = self._model(sample_inputs)

                    sample_targets_np = sample_targets.cpu().numpy()
                    sample_outputs_np = sample_outputs.cpu().numpy()
                    sample_inputs_np = sample_inputs.cpu().numpy()

                    if self._normalization_stats and 'targets' in self._normalization_stats:
                        target_stats = self._normalization_stats['targets']

                        self._logger.info(f"Estadísticas de normalización: {target_stats}")

                        tmins = target_stats.get('mins', 0)
                        tmaxs = target_stats.get('maxs', 1)
                        trange = tmaxs - tmins

                        if val_dataset._use_deltas:
                            sample_outputs_np = val_dataset.convert_deltas_to_absolutes(sample_outputs_np,
                                                                                        sample_inputs_np)
                            sample_targets_np = val_dataset.convert_deltas_to_absolutes(sample_targets_np,
                                                                                        sample_inputs_np)

                        denorm_targets = np.zeros_like(sample_targets_np)
                        denorm_outputs = np.zeros_like(sample_outputs_np)

                        for step in range(sample_targets_np.shape[1]):
                            denorm_targets[:, step, :] = sample_targets_np[:, step, :] * trange + tmins
                            denorm_outputs[:, step, :] = sample_outputs_np[:, step, :] * trange + tmins
                    else:
                        self._logger.warning("Couldn't find any normalization stats")
                        denorm_targets = sample_targets_np
                        denorm_outputs = sample_outputs_np
                        val_exp_var = explained_variance_score(all_val_targets, all_val_outputs)
                        val_nrmse = np.sqrt(val_mse) / (np.max(all_val_targets) - np.min(all_val_targets) + 1e-10)

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
                        step_mae = np.mean(np.abs(step_error))
                        self._logger.info(f"  MAE: {step_mae:.4f}")

                    total_mse = np.mean((denorm_outputs[example_idx] - denorm_targets[example_idx]) ** 2)
                    self._logger.info(f"MSE Total para todos los pasos: {total_mse:.4f}")
                    self._logger.info("=" * 50)

                self._logger.info(
                    f"Epoch {epoch + 1}/{hyperparameters['num_epochs']} - "
                    f"Time: {epoch_time:.1f}s - "
                    f"Train Loss (Huber): {avg_train_loss:.4f} - "
                    f"Val Loss (Huber): {avg_val_loss:.4f} - "
                    f"Train MSE: {train_mse:.4f} - "
                    f"Val MSE: {val_mse:.4f} - "
                    f"Train MAE: {train_mae:.4f} - "
                    f"Val MAE: {val_mae:.4f} - "
                    f"Train R²: {train_r2:.4f} - "
                    f"Val R²: {val_r2:.4f} - "
                    f"Train Accuracy: {train_accuracy:.2%} - "
                    f"Val Accuracy: {val_accuracy:.2%}"
                )
                self._logger.info(
                    f"Train ExpVar: {train_exp_var:.4f} - Val ExpVar: {val_exp_var:.4f} - "
                    f"Train NRMSE: {train_nrmse:.4f} - Val NRMSE: {val_nrmse:.4f}"
                )

            else:
                epoch_time = time.time() - start_time
                self._logger.info(
                    f"Epoch {epoch + 1}/{hyperparameters['num_epochs']} - "
                    f"Time: {epoch_time:.1f}s - "
                    f"Train Loss (Huber): {avg_train_loss:.4f} - "
                    f"Train MSE: {train_mse:.4f} - "
                    f"Train MAE: {train_mae:.4f} - "
                    f"Train R²: {train_r2:.4f} - "
                    f"Train Accuracy: {train_accuracy:.2%}"
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
            # self.analyze_correlation(data)

        return history

    def _calculate_accuracy(self, predictions, targets, absolute_tolerance=0.1, relative_tolerance=0.1):
        epsilon = 1e-10

        absolute_error = np.abs(predictions - targets)

        relative_error = absolute_error / (np.abs(targets) + epsilon)

        within_tolerance = (absolute_error <= absolute_tolerance) | (relative_error <= relative_tolerance)

        return np.mean(within_tolerance)

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
            prediction = self._model(sequence_tensor)

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
