"""
##########################################################################
#                                                                        #
#                           ✦ ECHOES OF GAIA ✦                           #
#                                                                        #
#    Trabajo Fin de Grado (TFG)                                          #
#    Facultad de Ingeniería Informática - Donostia                       #
#    UPV/EHU - Euskal Herriko Unibertsitatea                             #
#                                                                        #
#    Área de Computación e Inteligencia Artificial                       #
#                                                                        #
#    Autor:  Aingeru García Blas                                         #
#    GitHub: https://github.com/geru-scotland                            #
#    Repo:   https://github.com/geru-scotland/echoes-of-gaia             #
#                                                                        #
##########################################################################
"""
from enum import auto
from typing import TypedDict, NamedTuple, List, Optional, Dict, Literal, Any

from shared.enums.base import EnumBaseStr, EnumBaseInt


class BiomeType(EnumBaseStr):
    TROPICAL = "tropical"
    DESERT = "desert"
    TAIGA = "taiga"
    SAVANNA = "savanna"
    TUNDRA = "tundra"


# Maps
class TerrainType(EnumBaseInt):
    WATER_DEEP = 0
    WATER_MID = 1
    WATER_SHALLOW = 2
    SHORE = 3
    GRASS = 4
    MOUNTAIN = 5
    SNOW = 6
    SAND = 7


class ComponentType(EnumBaseStr):
    TRANSFORM = "transform"
    GROWTH = "growth"
    PHOTOSYNTHETIC_METABOLISM = "photosynthetic_metabolism"
    AUTOTROPHIC_NUTRITION = "autotrophic_nutrition"
    STATE = "state"
    VITAL = "vital"
    CLIMATE = "climate"
    WEATHER_ADAPTATION = "weather_adaptation"


class EntityState(EnumBaseStr):
    HEALTHY = "healthy"
    WEAK = "weak"
    DYING = "dying"


class EntityType(EnumBaseStr):
    FLORA = "flora"
    FAUNA = "fauna"
    HUMAN = "human"


class EntityIndex(EnumBaseInt):
    FLORA = 1
    FAUNA = 2


class FloraSpecies(EnumBaseStr):
    OAK_TREE = "oak_tree"
    BRAMBLE = "bramble"
    MUSHROOM = "mushroom"


class FaunaSpecies(EnumBaseStr):
    DEER = "deer"
    BOAR = "boar"
    FOX = "fox"


class Habitats:
    class Relations(EnumBaseStr):
        IN = "IN"
        NEARBY = "NEARBY"

    class Type(EnumBaseStr):
        FOREST = "forest"
        THICKET = "thicket"
        FUNGAL = "fungal_zone"
        COASTAL = "coastal"
        ALPINE = "alpine"


class CapturePeriod(EnumBaseInt):
    MONTHLY = 60
    WEEKLY = 15
    DAILY = 2
    CUSTOM = -1


class CaptureFormat(EnumBaseStr):
    JSON = "json"
    JSONL = "jsonl"
    MSGPACK_GZ = "msgpack.gz"
    YAML = "yaml"


class CaptureType(EnumBaseStr):
    FULL = "full"
    ENTITIES_ONLY = "entities_only"
    TERRAIN_ONLY = "terrain_only"
    METRICS_ONLY = "metrics_only"


class WeatherEvent(EnumBaseStr):
    CLEAR = "clear"
    SUNNY = "sunny"
    PARTLY_CLOUDY = "partly_cloudy"
    CLOUDY = "cloudy"
    FOG = "fog"
    DRIZZLE = "drizzle"
    RAIN = "rain"
    HEAVY_RAIN = "heavy_rain"
    THUNDERSTORM = "thunderstorm"
    LIGHT_SPRINKLE = "light_sprinkle"
    RAIN_SHOWER = "rain_shower"
    SNOW = "snow"
    BLIZZARD = "blizzard"
    HEATWAVE = "heatwave"
    DROUGHT = "drought"
    WINDY = "windy"


class Season(EnumBaseStr):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"


class FaunaAction(EnumBaseInt):
    MOVE_NORTH = auto()
    MOVE_SOUTH = auto()
    MOVE_EAST = auto()
    MOVE_WEST = auto()
    EAT = auto()
    DRINK = auto()
    REST = auto()
    HIDE = auto()
    FORAGE = auto()


class AgentType(EnumBaseStr):
    CLIMATE_AGENT = "climate_agent"
    EVOLUTION_AGENT = "evolution_agent"


class Agents:
    class Reinforcement(EnumBaseStr):
        NAIVE_CLIMATE = "climate"
        FAUNA = "fauna"


class MutationType(EnumBaseInt):
    ADAPTIVE = auto()
    GAUSSIAN = auto()


class TraitRecord(NamedTuple):
    generation: int
    value: float


class SpeciesTraitData(TypedDict):
    trait_name: str
    records: List[TraitRecord]
    trend: Optional[float]


class EvolutionSummary(TypedDict):
    key_adaptations: List[str]
    fitness_correlation: Dict[str, float]
    generations_tracked: int


class Hyperparams(TypedDict):
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    ent_coef: float
    verbose: int


class ModelConfig(TypedDict):
    algorithm: Literal["PPO", "DQN", "A2C", "SAC"]
    policy: str
    hyperparams: Hyperparams
    timesteps: int


class EnvConfig(TypedDict):
    env_class: str
    observation_space: Dict[str, Any]
    action_space: Dict[str, Any]


class ReinforcementConfig(TypedDict):
    name: str
    description: str
    version: str
    model: ModelConfig
    environment: EnvConfig
    output_path: str
