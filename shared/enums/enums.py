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
from enum import auto, Enum
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
    UNKNWON = 8


class ComponentType(EnumBaseStr):
    TRANSFORM = "transform"
    GROWTH = "growth"
    PHOTOSYNTHETIC_METABOLISM = "photosynthetic_metabolism"
    AUTOTROPHIC_NUTRITION = "autotrophic_nutrition"
    HETEROTROPHIC_NUTRITION = "heterotrophic_nutrition"
    STATE = "state"
    VITAL = "vital"
    CLIMATE = "climate"
    WEATHER_ADAPTATION = "weather_adaptation"
    MOVEMENT = "movement"


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
    HELICONIA = "heliconia"
    MANGROVE = "mangrove"
    ORCHID = "orchid"
    BANANA_TREE = "banana_tree"
    CACTUS = "cactus"
    DATE_PALM = "date_palm"
    DESERT_SAGE = "desert_sage"
    PRICKLY_PEAR = "prickly_pear"
    PINE_TREE = "pine_tree"
    ARCTIC_MOSS = "arctic_moss"
    SPRUCE_TREE = "spruce_tree"
    WINTERBERRY = "winterberry"
    ACACIA_TREE = "acacia_tree"
    SAVANNA_GRASS = "savanna_grass"
    BAOBAB_TREE = "baobab_tree"
    ALOE_VERA = "aloe_vera"
    ARCTIC_WILLOW = "arctic_willow"
    TUNDRA_LICHEN = "tundra_lichen"
    ARCTIC_COTTON_GRASS = "arctic_cotton_grass"
    SAXIFRAGE = "saxifrage"


class FaunaSpecies(EnumBaseStr):
    DEER = "deer"
    BOAR = "boar"
    FOX = "fox"
    PANTHER = "panther"
    TOUCAN = "toucan"
    CAPYBARA = "capybara"
    JAGUAR = "jaguar"
    POISON_DART_FROG = "poison_dart_frog"
    DESERT_FOX = "desert_fox"
    SCORPION = "scorpion"
    CAMEL = "camel"
    DESERT_LIZARD = "desert_lizard"
    WOLF = "wolf"
    MOOSE = "moose"
    LYNX = "lynx"
    SNOWSHOE_HARE = "snowshoe_hare"
    LION = "lion"
    ZEBRA = "zebra"
    ELEPHANT = "elephant"
    HYENA = "hyena"
    POLAR_BEAR = "polar_bear"
    ARCTIC_FOX = "arctic_fox"
    MUSK_OX = "musk_ox"
    ARCTIC_HARE = "arctic_hare"


class Habitats:
    class Relations(EnumBaseStr):
        IN = "IN"
        NEARBY = "NEARBY"

    class Type(EnumBaseStr):
        FOREST = "forest"
        THICKET = "thicket"
        FUNGAL_ZONE = "fungal_zone"
        COASTAL = "coastal"
        ALPINE = "alpine"
        RAINFOREST_CANOPY = "rainforest_canopy"
        TROPICAL_WETLAND = "tropical_wetland"
        TROPICAL_UNDERSTORY = "tropical_understory"
        SAND_DUNES = "sand_dunes"
        ROCKY_DESERT = "rocky_desert"
        DESERT_OASIS = "desert_oasis"
        CONIFEROUS_FOREST = "coniferous_forest"
        TAIGA_CLEARING = "taiga_clearing"
        ROCKY_TAIGA = "rocky_taiga"
        TAIGA_FOREST_EDGE = "taiga_forest_edge"
        TAIGA_WETLAND = "taiga_wetland"
        TAIGA_UNDERGROWTH = "taiga_undergrowth"
        GRASSLAND = "grassland"
        SAVANNA_WOODLAND = "savanna_woodland"
        SEASONAL_RIVER = "seasonal_river"
        PERMAFROST_PLAINS = "permafrost_plains"
        ALPINE_SLOPES = "alpine_slopes"
        TUNDRA_LAKE = "tundra_lake"


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
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3

    def __str__(self):
        return str(self.value)


class AgentType(EnumBaseStr):
    CLIMATE_AGENT = "climate_agent"
    EVOLUTION_AGENT = "evolution_agent"
    FAUNA_AGENT = "fauna_agent"
    EQUILIBRIUM_AGENT = "equilibrium"


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


class LocalFovConfig(TypedDict):
    size: Dict[str, Any]
    center: int


class ReinforcementConfig(TypedDict):
    name: str
    description: str
    version: str
    model: ModelConfig
    environment: EnvConfig
    output_path: str
    local_fov: LocalFovConfig


class SimulationMode(EnumBaseInt):
    NORMAL = auto()
    TRAINING = auto()
    TRAINING_WITH_RL_MODEL = auto()
    TRAINING_FINISHED = auto()
    UNTIL_EXTINCTION = auto()


class NeuralMode(EnumBaseInt):
    TRAINING = auto()
    INFERENCE = auto()


class Direction(Enum):
    NORTH = (-1, 0)
    SOUTH = (1, 0)
    EAST = (0, 1)
    WEST = (0, -1)


class PositionNotValidReason(EnumBaseInt):
    POSITION_OUT_OF_BOUNDARIES = 0
    POSITION_BUSY = 1
    POSITION_NON_TRAVERSABLE = 2
    NONE = 3


class DietType(Enum):
    HERBIVORE = "herbivore"
    CARNIVORE = "carnivore"
    OMNIVORE = "omnivore"


from enum import Enum


class SpeciesStatus(str, Enum):
    ENDANGERED = "endangered"
    STRESSED = "stressed"
    OVERPOPULATED = "overpopulated"


class SpeciesAction(str, Enum):
    PROTECTION_NEEDED = "protection_needed"
    STRESS_REDUCTION_NEEDED = "stress_reduction_needed"
    POPULATION_CONTROL_NEEDED = "population_control_needed"


class PredatorPreyBalance(str, Enum):
    PREDATOR_DOMINANT = "predator_dominant"
    PREY_DOMINANT = "prey_dominant"


class EcosystemRisk(str, Enum):
    PREY_EXTINCTION_RISK = "prey_extinction_risk"
    OVERPOPULATION_RISK = "overpopulation_risk"


class RecommendedAction(str, Enum):
    REDUCE_PREDATOR_PRESSURE = "reduce_predator_pressure"
    INCREASE_PREDATOR_PRESSURE = "increase_predator_pressure"
    INTRODUCE_SPECIES_DIVERSITY = "introduce_species_diversity"
    ECOSYSTEM_INTERVENTION_NEEDED = "ecosystem_intervention_needed"


class BiodiversityStatus(str, Enum):
    CRITICAL = "critical"


class StabilityStatus(str, Enum):
    UNSTABLE = "unstable"
    HIGHLY_STABLE = "highly_stable"


class InterventionPriority(str, Enum):
    LOW = "low"


class ClimateStatus(str, Enum):
    EXTREME_COLD = "extreme_cold"
    EXTREME_HEAT = "extreme_heat"


class ClimateAction(str, Enum):
    INCREASE_TEMPERATURE = "increase_temperature"
    REDUCE_TEMPERATURE = "reduce_temperature"


class MoistureStatus(str, Enum):
    EXTREME_DRY = "extreme_dry"
    EXTREME_WET = "extreme_wet"


class MoistureAction(str, Enum):
    INCREASE_HUMIDITY = "increase_humidity"
    REDUCE_PRECIPITATION = "reduce_precipitation"
