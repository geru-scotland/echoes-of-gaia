import pytest

from shared.stores.biome_store import BiomeStore
from biome.systems.maps.procedural_maps import MapGenerator, Map, PerlinNoiseGenerator
from shared.constants import MAP_DEFAULT_SIZE

@pytest.fixture
def bioma_store_load():
    BiomeStore.load_ecosystem_data()

def test_map_generation(bioma_store_load):

    map_data = {
        "size": MAP_DEFAULT_SIZE,
        "weights": [40, 10, 10, 12, 55, 30, 0]
    }

    generator = MapGenerator(PerlinNoiseGenerator)
    generated_map = generator.generate(map_data=map_data)

    assert isinstance(generated_map, Map), "The object is not an instance from the Map class"
    assert len(generated_map.tile_map) == MAP_DEFAULT_SIZE[1]+1, "The height is not correct"
    assert len(generated_map.tile_map[0]) == MAP_DEFAULT_SIZE[0]+1, "The width is not correct"

    assert all(
        -1.0 <= noise_value <= 1.0
        for row in generated_map.noise_map
        for noise_value in row
    ), "The values within the noise map are not within the expected range (-1.0 a 1.0)"

    valid_terrain_types = BiomeStore.terrains
    assert all(
        tile in valid_terrain_types
        for row in generated_map.tile_map
        for tile in row
    ), "Tile map contains incorrect values"
