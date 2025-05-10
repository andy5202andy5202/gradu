from server_definition import EdgeServer

def init_edge_servers(cached_node_data,DATA_PATH, active_training_threads, global_server, global_clock, upload_due_to_position):
    edge0 = EdgeServer('Edge0', covered_edges={
        'n_0_4_n_0_5',
        'n_0_5_n_0_4',
        'n_0_5_n_0_6',
        'n_0_6_n_0_5',
        'n_0_6_n_1_6',
        'n_1_6_n_0_6',
        'n_1_6_n_2_6',
        'n_2_6_n_1_6',
        'n_2_6_n_2_5',
        'n_2_5_n_2_4',
        'n_2_4_n_1_4',
        'n_1_4_n_0_4',
        'n_0_5_n_1_5',
        'n_1_5_n_0_5',
        'n_2_5_n_1_5',
        'n_1_5_n_2_5',
        'n_1_6_n_1_5',
        'n_1_5_n_1_6',
        'n_1_5_n_1_4',
        'n_1_4_n_1_5'
    },cached_node_data=cached_node_data, global_data_path=DATA_PATH, active_training_threads=active_training_threads, global_server=global_server, global_clock=global_clock, upload_due_to_position=upload_due_to_position
)  # patched

    edge1 = EdgeServer('Edge1', covered_edges={
        'n_2_4_n_2_5',
        'n_2_5_n_2_6',
        'n_2_6_n_3_6',
        'n_3_6_n_2_6',
        'n_3_6_n_4_6',
        'n_4_6_n_3_6',
        'n_4_6_n_4_5',
        'n_4_5_n_4_4',
        'n_4_4_n_3_4',
        'n_3_4_n_2_4',
        'n_2_5_n_3_5',
        'n_3_5_n_2_5',
        'n_3_5_n_4_5',
        'n_4_5_n_3_5',
        'n_3_5_n_3_6',
        'n_3_6_n_3_5',
        'n_3_4_n_3_5',
        'n_3_5_n_3_4'
    },cached_node_data=cached_node_data, global_data_path=DATA_PATH, active_training_threads=active_training_threads, global_server=global_server, global_clock=global_clock, upload_due_to_position=upload_due_to_position
)  # patched

    edge2 = EdgeServer('Edge2', covered_edges={
        'n_4_4_n_4_5',
        'n_4_5_n_4_6',
        'n_4_6_n_5_6',
        'n_5_6_n_4_6',
        'n_5_6_n_6_6',
        'n_6_6_n_5_6',
        'n_6_6_n_6_5',
        'n_6_5_n_6_6',
        'n_6_4_n_6_5',
        'n_6_5_n_6_4',
        'n_6_4_n_5_4',
        'n_5_4_n_4_4',
        'n_4_5_n_5_5',
        'n_5_5_n_4_5',
        'n_5_5_n_6_5',
        'n_6_5_n_5_5',
        'n_5_5_n_5_6',
        'n_5_6_n_5_5',
        'n_5_5_n_5_4',
        'n_5_4_n_5_5'
    },cached_node_data=cached_node_data, global_data_path=DATA_PATH, active_training_threads=active_training_threads, global_server=global_server, global_clock=global_clock, upload_due_to_position=upload_due_to_position

)  # patched

    edge3 = EdgeServer('Edge3', covered_edges={
        'n_0_2_n_0_3',
        'n_0_3_n_0_2',
        'n_0_3_n_0_4',
        'n_0_4_n_0_3',
        'n_0_4_n_1_4',
        'n_1_4_n_2_4',
        'n_2_4_n_2_3',
        'n_2_3_n_2_2',
        'n_1_2_n_0_2',
        'n_2_2_n_1_2',
        'n_0_3_n_1_3',
        'n_1_3_n_0_3',
        'n_1_3_n_2_3',
        'n_2_3_n_1_3',
        'n_1_3_n_1_4',
        'n_1_4_n_1_3',
        'n_1_3_n_1_2',
        'n_1_2_n_1_3'
    },cached_node_data=cached_node_data, global_data_path=DATA_PATH, active_training_threads=active_training_threads, global_server=global_server, global_clock=global_clock, upload_due_to_position=upload_due_to_position

)  # patched

    edge4 = EdgeServer('Edge4', covered_edges={
        'n_2_2_n_2_3',
        'n_2_3_n_2_4',
        'n_2_4_n_3_4',
        'n_3_4_n_4_4',
        'n_4_4_n_4_3',
        'n_4_3_n_4_2',
        'n_4_2_n_3_2',
        'n_3_2_n_2_2',
        'n_2_3_n_3_3',
        'n_3_3_n_2_3',
        'n_3_3_n_4_3',
        'n_4_3_n_3_3',
        'n_3_3_n_3_4',
        'n_3_4_n_3_3',
        'n_3_2_n_3_3',
        'n_3_3_n_3_2'
    },cached_node_data=cached_node_data, global_data_path=DATA_PATH, active_training_threads=active_training_threads, global_server=global_server, global_clock=global_clock, upload_due_to_position=upload_due_to_position

)  # patched

    edge5 = EdgeServer('Edge5', covered_edges={
        'n_4_2_n_4_3',
        'n_4_3_n_4_4',
        'n_4_4_n_5_4',
        'n_5_4_n_6_4',
        'n_6_3_n_6_4',
        'n_6_4_n_6_3',
        'n_6_2_n_6_3',
        'n_6_3_n_6_2',
        'n_5_2_n_4_2',
        'n_6_2_n_5_2',
        'n_5_3_n_4_3',
        'n_4_3_n_5_3',
        'n_5_3_n_6_3',
        'n_6_3_n_5_3',
        'n_5_3_n_5_4',
        'n_5_4_n_5_3',
        'n_5_2_n_5_3',
        'n_5_3_n_5_2'
    },cached_node_data=cached_node_data, global_data_path=DATA_PATH, active_training_threads=active_training_threads, global_server=global_server, global_clock=global_clock, upload_due_to_position=upload_due_to_position

)  # patched

    edge6 = EdgeServer('Edge6', covered_edges={
        'n_0_0_n_0_1',
        'n_0_1_n_0_0',
        'n_0_2_n_0_1',
        'n_0_1_n_0_2',
        'n_0_2_n_1_2',
        'n_1_2_n_2_2',
        'n_2_2_n_2_1',
        'n_2_1_n_2_0',
        'n_1_0_n_2_0',
        'n_2_0_n_1_0',
        'n_0_0_n_1_0',
        'n_1_0_n_0_0',
        'n_0_1_n_1_1',
        'n_1_1_n_0_1',
        'n_1_1_n_2_1',
        'n_2_1_n_1_1',
        'n_1_2_n_1_1',
        'n_1_1_n_1_2',
        'n_1_1_n_1_0',
        'n_1_0_n_1_1'
    },cached_node_data=cached_node_data, global_data_path=DATA_PATH, active_training_threads=active_training_threads, global_server=global_server, global_clock=global_clock, upload_due_to_position=upload_due_to_position

)  # patched

    edge7 = EdgeServer('Edge7', covered_edges={
        'n_2_0_n_2_1',
        'n_2_1_n_2_2',
        'n_2_2_n_3_2',
        'n_3_2_n_4_2',
        'n_4_2_n_4_1',
        'n_4_1_n_4_0',
        'n_3_0_n_4_0',
        'n_4_0_n_3_0',
        'n_3_0_n_2_0',
        'n_2_0_n_3_0',
        'n_3_1_n_2_1',
        'n_2_1_n_3_1',
        'n_3_1_n_4_1',
        'n_4_1_n_3_1',
        'n_3_1_n_3_2',
        'n_3_2_n_3_1',
        'n_3_1_n_3_0',
        'n_3_0_n_3_1'
    },cached_node_data=cached_node_data, global_data_path=DATA_PATH, active_training_threads=active_training_threads, global_server=global_server, global_clock=global_clock, upload_due_to_position=upload_due_to_position

)  # patched

    edge8 = EdgeServer('Edge8', covered_edges={
        'n_4_0_n_4_1',
        'n_4_1_n_4_2',
        'n_4_2_n_5_2',
        'n_5_2_n_6_2',
        'n_6_1_n_6_2',
        'n_6_2_n_6_1',
        'n_6_1_n_6_0',
        'n_6_0_n_6_1',
        'n_5_0_n_6_0',
        'n_6_0_n_5_0',
        'n_5_0_n_4_0',
        'n_4_0_n_5_0',
        'n_5_1_n_4_1',
        'n_4_1_n_5_1',
        'n_5_1_n_6_1',
        'n_6_1_n_5_1',
        'n_5_1_n_5_2',
        'n_5_2_n_5_1',
        'n_5_1_n_5_0',
        'n_5_0_n_5_1'
    },cached_node_data=cached_node_data, global_data_path=DATA_PATH, active_training_threads=active_training_threads, global_server=global_server, global_clock=global_clock, upload_due_to_position=upload_due_to_position

)  # patched

    return {
        'Edge0': edge0,
        'Edge1': edge1,
        'Edge2': edge2,
        'Edge3': edge3,
        'Edge4': edge4,
        'Edge5': edge5,
        'Edge6': edge6,
        'Edge7': edge7,
        'Edge8': edge8,
    }




# edge0 = EdgeServer('Edge0', 
#     covered_edges={
#         'n_0_4_n_0_5',
#         'n_0_5_n_0_4',
#         'n_0_5_n_0_6',
#         'n_0_6_n_0_5',
#         'n_0_6_n_1_6',
#         'n_1_6_n_0_6',
#         'n_1_6_n_2_6',
#         'n_2_6_n_1_6',
#         'n_2_6_n_2_5',
#         'n_2_5_n_2_4',
#         'n_2_4_n_1_4',
#         'n_1_4_n_0_4',
#         'n_0_5_n_1_5',
#         'n_1_5_n_0_5',
#         'n_2_5_n_1_5',
#         'n_1_5_n_2_5',
#         'n_1_6_n_1_5',
#         'n_1_5_n_1_6',
#         'n_1_5_n_1_4',
#         'n_1_4_n_1_5'
#     }
# )

# edge1 = EdgeServer('Edge1', covered_edges={
#     'n_2_4_n_2_5',
#     'n_2_5_n_2_6',
#     'n_2_6_n_3_6',
#     'n_3_6_n_2_6',
#     'n_3_6_n_4_6',
#     'n_4_6_n_3_6',
#     'n_4_6_n_4_5',
#     'n_4_5_n_4_4',
#     'n_4_4_n_3_4',
#     'n_3_4_n_2_4',
#     'n_2_5_n_3_5',
#     'n_3_5_n_2_5',
#     'n_3_5_n_4_5',
#     'n_4_5_n_3_5',
#     'n_3_5_n_3_6',
#     'n_3_6_n_3_5',
#     'n_3_4_n_3_5',
#     'n_3_5_n_3_4',
# })

# edge2 = EdgeServer('Edge2', covered_edges={
#     'n_4_4_n_4_5',
#     'n_4_5_n_4_6',
#     'n_4_6_n_5_6',
#     'n_5_6_n_4_6',
#     'n_5_6_n_6_6',
#     'n_6_6_n_5_6',
#     'n_6_6_n_6_5',
#     'n_6_5_n_6_6',
#     'n_6_4_n_6_5',
#     'n_6_5_n_6_4',
#     'n_6_4_n_5_4',
#     'n_5_4_n_4_4',
#     'n_4_5_n_5_5',
#     'n_5_5_n_4_5',
#     'n_5_5_n_6_5',
#     'n_6_5_n_5_5',
#     'n_5_5_n_5_6',
#     'n_5_6_n_5_5',
#     'n_5_5_n_5_4',
#     'n_5_4_n_5_5'
# })

# edge3 = EdgeServer('Edge3', covered_edges={
#     'n_0_2_n_0_3',
#     'n_0_3_n_0_2',
#     'n_0_3_n_0_4',
#     'n_0_4_n_0_3',
#     'n_0_4_n_1_4',
#     'n_1_4_n_2_4',
#     'n_2_4_n_2_3',
#     'n_2_3_n_2_2',
#     'n_1_2_n_0_2',
#     'n_2_2_n_1_2',
#     'n_0_3_n_1_3',
#     'n_1_3_n_0_3',
#     'n_1_3_n_2_3',
#     'n_2_3_n_1_3',
#     'n_1_3_n_1_4',
#     'n_1_4_n_1_3',
#     'n_1_3_n_1_2',
#     'n_1_2_n_1_3',
# })

# edge4 = EdgeServer('Edge4', covered_edges={
#     'n_2_2_n_2_3',
#     'n_2_3_n_2_4',
#     'n_2_4_n_3_4',
#     'n_3_4_n_4_4',
#     'n_4_4_n_4_3',
#     'n_4_3_n_4_2',
#     'n_4_2_n_3_2',
#     'n_3_2_n_2_2',
#     'n_2_3_n_3_3',
#     'n_3_3_n_2_3',
#     'n_3_3_n_4_3',
#     'n_4_3_n_3_3',
#     'n_3_3_n_3_4',
#     'n_3_4_n_3_3',
#     'n_3_2_n_3_3',
#     'n_3_3_n_3_2',
# })

# edge5 = EdgeServer('Edge5', covered_edges={
#     'n_4_2_n_4_3',
#     'n_4_3_n_4_4',
#     'n_4_4_n_5_4',
#     'n_5_4_n_6_4',
#     'n_6_3_n_6_4',
#     'n_6_4_n_6_3',
#     'n_6_2_n_6_3',
#     'n_6_3_n_6_2',
#     'n_5_2_n_4_2',
#     'n_6_2_n_5_2',
#     'n_5_3_n_4_3',
#     'n_4_3_n_5_3',
#     'n_5_3_n_6_3',
#     'n_6_3_n_5_3',
#     'n_5_3_n_5_4',
#     'n_5_4_n_5_3',
#     'n_5_2_n_5_3',
#     'n_5_3_n_5_2',
# })

# edge6 = EdgeServer('Edge6', covered_edges={
#     'n_0_0_n_0_1',
#     'n_0_1_n_0_0',
#     'n_0_2_n_0_1',
#     'n_0_1_n_0_2',
#     'n_0_2_n_1_2',
#     'n_1_2_n_2_2',
#     'n_2_2_n_2_1',
#     'n_2_1_n_2_0',
#     'n_1_0_n_2_0',
#     'n_2_0_n_1_0',
#     'n_0_0_n_1_0',
#     'n_1_0_n_0_0',
#     'n_0_1_n_1_1',
#     'n_1_1_n_0_1',
#     'n_1_1_n_2_1',
#     'n_2_1_n_1_1',
#     'n_1_2_n_1_1',
#     'n_1_1_n_1_2',
#     'n_1_1_n_1_0',
#     'n_1_0_n_1_1'
# })

# edge7 = EdgeServer('Edge7', covered_edges={
#     'n_2_0_n_2_1',
#     'n_2_1_n_2_2',
#     'n_2_2_n_3_2',
#     'n_3_2_n_4_2',
#     'n_4_2_n_4_1',
#     'n_4_1_n_4_0',
#     'n_3_0_n_4_0',
#     'n_4_0_n_3_0',
#     'n_3_0_n_2_0',
#     'n_2_0_n_3_0',
#     'n_3_1_n_2_1',
#     'n_2_1_n_3_1',
#     'n_3_1_n_4_1',
#     'n_4_1_n_3_1',
#     'n_3_1_n_3_2',
#     'n_3_2_n_3_1',
#     'n_3_1_n_3_0',
#     'n_3_0_n_3_1'
# })

# edge8 = EdgeServer('Edge8', covered_edges={
#     'n_4_0_n_4_1',
#     'n_4_1_n_4_2',
#     'n_4_2_n_5_2',
#     'n_5_2_n_6_2',
#     'n_6_1_n_6_2',
#     'n_6_2_n_6_1',
#     'n_6_1_n_6_0',
#     'n_6_0_n_6_1',
#     'n_5_0_n_6_0',
#     'n_6_0_n_5_0',
#     'n_5_0_n_4_0',
#     'n_4_0_n_5_0',
#     'n_5_1_n_4_1',
#     'n_4_1_n_5_1',
#     'n_5_1_n_6_1',
#     'n_6_1_n_5_1',
#     'n_5_1_n_5_2',
#     'n_5_2_n_5_1',
#     'n_5_1_n_5_0',
#     'n_5_0_n_5_1'
# })

# edge_servers = {
#     'Edge0': edge0,
#     'Edge1': edge1,
#     'Edge2': edge2,
#     'Edge3': edge3,
#     'Edge4': edge4,
#     'Edge5': edge5,
#     'Edge6': edge6,
#     'Edge7': edge7,
#     'Edge8': edge8,
# }
