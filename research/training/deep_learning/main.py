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
Entry point for deep learning model training in biome simulations.

Provides command-line interface for training neural time-series models; handles
configuration loading and training session initialization. Serves as the main
execution script for deep learning pipeline - reports training outcomes
and validation metrics after completion.
"""

import os
import sys
import logging
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from research.training.deep_learning.model_manager import NeuralModelManager
from shared.enums.strings import Loggers
from utils.loggers import LoggerManager
from utils.paths import BASE_DIR


def main():
    logger: logging.Logger = LoggerManager.get_logger(Loggers.DEEP_LEARNING)
    parser = argparse.ArgumentParser(description='Train LSTM model for neurosymbolic system')
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()

    config_path = args.config

    trainer = NeuralModelManager(config_path)

    logger.info("Starting training")
    history = trainer.train()

    logger.info("Training completed")
    logger.info(f"Final training loss: {history['train_loss'][-1]:.4f}")
    if 'val_loss' in history and history['val_loss']:
        logger.info(f"Final validation loss: {history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
