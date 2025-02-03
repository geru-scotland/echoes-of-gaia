from simulation.api.simulation_api import SimulationAPI

print("Welcome to the Research Hub")
simulation_api: SimulationAPI = SimulationAPI()
simulation_api.initialise()
simulation_api.run()