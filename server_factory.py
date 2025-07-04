import os
import multiprocessing
from server_definition_rl import GlobalServer
from edge_server_init import init_edge_servers
from vehicle_data_collection import preload_blurred_data  
from global_clock import GlobalClock
from vehicle_manager_rl import init_environment

# server_factory.py

def create_servers_fn():
    return init_environment()