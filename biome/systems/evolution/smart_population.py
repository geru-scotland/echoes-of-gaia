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
import itertools
from typing import List, Optional, Dict, Any, Tuple

import skfuzzy as fuzz
from skfuzzy import control
import numpy as np

from shared.enums.events import SimulationEvent
from shared.events.handler import EventHandler
from simulation.core.systems.events.event_bus import SimulationEventBus


class SmartPopulationTrendControl(EventHandler):
    def __init__(self, species_name, base_lifespan, logger):
        super().__init__()
        self._species_name = species_name
        self._base_lifespan = base_lifespan
        self._logger = logger
        self._generations_history: List[int] = []
        self._population_history: List[int] = []
        self._slope_history: List[float] = []
        self._adjustments_history: List[Tuple[float, float]] = []
        self._last_fuzzy_data: Optional[Dict[str, Any]] = {}
        self._max_history_length = 100
        self._generation_counter = itertools.count(0)

    def _register_events(self):
            SimulationEventBus.register(SimulationEvent.SIMULATION_FINISHED, self._handle_simulation_finished)

    def _handle_simulation_finished(self):
        import matplotlib.pyplot as plt
        self._logger.warning(f"Fuzzy logic adjustment history: {self._adjustments_history}")
        if len(self._population_history) > 1:
            plt.figure(figsize=(10, 6))

            plt.plot(self._generations_history, self._population_history, 'o-', color='blue',
                     markersize=4, label='Actual Population')

            x = np.array(self._generations_history)
            y = np.array(self._population_history)

            slope, intercept = np.polyfit(x, y, 1)
            trend_line = slope * x + intercept
            plt.plot(x, trend_line, '--', color='red',
                     label=f'Trend (slope={slope:.2f})')

            if len(self._generations_history) > 0:
                last_gen = self._generations_history[-1]
                future_gens = np.array([last_gen + i for i in range(1, 4)])
                predictions = slope * future_gens + intercept
                plt.plot(future_gens, predictions, 'o--', color='green',
                         label='Predictions')

            if len(self._population_history) > 0:
                current_pop = self._population_history[-1]
                avg_pop = np.mean(self._population_history)
                min_pop = min(self._population_history)
                max_pop = max(self._population_history)

                if current_pop > 0:
                    normalized_slope = slope / current_pop
                else:
                    normalized_slope = 0

                info_text = (
                    f"Current Pop: {current_pop}\n"
                    f"Average: {avg_pop:.1f}\n"
                    f"Min/Max: {min_pop}/{max_pop}\n"
                    f"Trend: {'↑' if slope > 0 else '↓'} {abs(slope):.2f} per gen\n"
                    f"Norm. Slope: {normalized_slope:.4f}"
                )

                plt.text(0.02, 0.95, info_text, transform=plt.gca().transAxes,
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                         verticalalignment='top')

            plt.title(f'Population Trend for {self._species_name}')
            plt.xlabel('Generation')
            plt.ylabel('Population')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

            plt.show()

            if self._last_fuzzy_data:
                try:
                    return
                    plt.figure(figsize=(12, 10))

                    plt.subplot(2, 1, 1)
                    for name, mf in self._last_fuzzy_data['slope_input'].terms.items():
                        plt.plot(self._last_fuzzy_data['slope_input'].universe,
                                 mf.mf,
                                 label=name)

                    plt.axvline(x=self._last_fuzzy_data['normalized_slope'],
                                color='black', linestyle='--',
                                label=f"Current: {self._last_fuzzy_data['normalized_slope']:.4f}")
                    plt.legend()
                    plt.title('Fuzzy Membership - Population Slope')
                    plt.ylabel('Membership')
                    plt.xlabel('Normalized Slope')
                    plt.grid(True)

                    plt.subplot(2, 1, 2)
                    for name, mf in self._last_fuzzy_data['adjustment_output'].terms.items():
                        plt.plot(self._last_fuzzy_data['adjustment_output'].universe,
                                 mf.mf,
                                 label=name)

                    plt.axvline(x=self._last_fuzzy_data['adjustment_factor'],
                                color='black', linestyle='--',
                                label=f"Result: {self._last_fuzzy_data['adjustment_factor']:.4f}")
                    plt.legend()
                    plt.title('Fuzzy Membership - Adjustment Factor')
                    plt.ylabel('Membership')
                    plt.xlabel('Adjustment')
                    plt.grid(True)

                    plt.tight_layout()
                    plt.show()

                except Exception as e:
                    self._logger.warning(f"Could not generate fuzzy visualization: {str(e)}")

    def record_population(self, current_population: int) -> None:
        self._population_history.append(current_population)
        self._generations_history.append(next(self._generation_counter))
        if len(self._population_history) > self._max_history_length:
            self._population_history = self._population_history[-self._max_history_length:]

    def calculate_adjustment(self) -> float:
        if len(self._population_history) < 2:
            return 1.0

        x = np.array([generation for generation in self._generations_history])
        y = np.array([population for population in self._population_history])

        slope = np.polyfit(x, y, deg=1)[0]
        # Con pendiente de recta, ahora calculo la tasa de cambio en el punto de la generación actual.
        current_population = self._population_history[-1]

        if current_population > 0:
            normalized_slope = slope / current_population
        else:
            normalized_slope = 0

        self._slope_history.append(normalized_slope)
        self._logger.warning(
            f"Slope normalized for {self._species_name}: {slope:.2f} (normalized: {normalized_slope:.2f})")


        slope_input = control.Antecedent(np.linspace(-1, 1, 100), 'slope')
        adjustment = control.Consequent(np.linspace(0.5, 2.5, 100), 'adjustment')
        # Nota: trapmf: rango plateau, todos los valores se consideran pertenencia completa 1.
        # Es decir, si está en este rango, vale ese adjustement.
        # Y el trimf: quiero un valor central y conforme me alejo , que decrezca linealmente
        # Fuzzy logic system setup - ADJUSTED FOR YOUR DATA RANGE

        slope_input['severe_decline'] = fuzz.trapmf(slope_input.universe, [-0.15, -0.12, -0.09, -0.07])
        slope_input['moderate_decline'] = fuzz.trimf(slope_input.universe, [-0.08, -0.04, -0.01])
        slope_input['stable'] = fuzz.trimf(slope_input.universe, [-0.02, 0.01, 0.04])
        slope_input['moderate_growth'] = fuzz.trimf(slope_input.universe, [0.03, 0.07, 0.12])
        slope_input['rapid_growth'] = fuzz.trapmf(slope_input.universe, [0.1, 0.15, 0.25, 0.3])

        adjustment['large_increase'] = fuzz.trimf(adjustment.universe, [1.8, 2.2, 2.5])
        adjustment['moderate_increase'] = fuzz.trimf(adjustment.universe, [1.4, 1.6, 1.9])
        adjustment['small_increase'] = fuzz.trimf(adjustment.universe, [1.1, 1.2, 1.4])
        adjustment['neutral'] = fuzz.trimf(adjustment.universe, [0.95, 1.0, 1.05])
        adjustment['reduction'] = fuzz.trimf(adjustment.universe, [0.5, 0.7, 0.95])

        rule1 = control.Rule(slope_input['severe_decline'], adjustment['large_increase'])
        rule2 = control.Rule(slope_input['moderate_decline'], adjustment['moderate_increase'])
        rule3 = control.Rule(slope_input['stable'], adjustment['small_increase'])
        rule4 = control.Rule(slope_input['moderate_growth'], adjustment['small_increase'])
        rule5 = control.Rule(slope_input['rapid_growth'], adjustment['reduction'])

        control_system = control.ControlSystem([rule1, rule2, rule3, rule4, rule5])
        simulator = control.ControlSystemSimulation(control_system)

        normalized_slope = max(min(normalized_slope, 0.3), -0.15)
        simulator.input['slope'] = normalized_slope


        try:
            simulator.compute()
            adjustment_factor = simulator.output["adjustment"]

            self._logger.info(f"Adjustment factor (fuzzy logic): {adjustment_factor:.4f}")
            self._logger.info(f"Normalized slope: {normalized_slope:.4f}")

            self._adjustments_history.append((normalized_slope, adjustment_factor))

            self._last_fuzzy_data = {
                'normalized_slope': normalized_slope,
                'adjustment_factor': adjustment_factor,
                'slope_input': slope_input,
                'adjustment_output': adjustment,
                'rules': [rule1, rule2, rule3, rule4, rule5]
            }

            return adjustment_factor
        except Exception as e:
            self._logger.warning(f"Error in fuzzy calculation: {str(e)}.")

    def predict_future_population(self, generations_ahead: int = 1) -> int:
        if len(self._population_history) < 2:
            return self._population_history[-1] if self._population_history else 0

        x = np.array([generation for generation in self._generations_history])
        y = np.array([population for population in self._population_history])

        slope, intercept = np.polyfit(x, y, deg=1)

        next_generation = self._generations_history[-1] + generations_ahead - 1

        predicted_population = slope * next_generation + intercept

        return max(0, int(round(predicted_population)))