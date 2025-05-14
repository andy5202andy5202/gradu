import os
import subprocess
import random

# ----------------------------
# Configuration for 7x7 Grid
# ----------------------------
GRID_SIZE = 7
GRID_SPACING = 200  # 每格距離（可調整）
NODES_FILE = "grid7x7.nod.xml"
EDGES_FILE = "grid7x7.edg.xml"
NET_FILE = "grid7x7.net.xml"
SUMOCFG_FILE = "grid7x7.sumocfg"
ROUTES_FILE = "grid7x7.rou.xml"

# ----------------------------
# Generate Nodes
# ----------------------------
with open(NODES_FILE, 'w') as f:
    f.write('<nodes>\n')
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            node_id = f"n_{x}_{y}"
            f.write(f'    <node id="{node_id}" x="{x * GRID_SPACING}" y="{y * GRID_SPACING}" type="priority"/>\n')
    f.write('</nodes>\n')

# ----------------------------
# Generate Edges (bidirectional)
# ----------------------------
with open(EDGES_FILE, 'w') as f:
    f.write('<edges>\n')
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            node_id = f"n_{x}_{y}"
            if x < GRID_SIZE - 1:
                right_node = f"n_{x+1}_{y}"
                f.write(f'    <edge id="{node_id}_{right_node}" from="{node_id}" to="{right_node}" numLanes="2" speed="20.0"/>\n')
                f.write(f'    <edge id="{right_node}_{node_id}" from="{right_node}" to="{node_id}" numLanes="2" speed="20.0"/>\n')
            if y < GRID_SIZE - 1:
                top_node = f"n_{x}_{y+1}"
                f.write(f'    <edge id="{node_id}_{top_node}" from="{node_id}" to="{top_node}" numLanes="2" speed="20.0"/>\n')
                f.write(f'    <edge id="{top_node}_{node_id}" from="{top_node}" to="{node_id}" numLanes="2" speed="20.0"/>\n')
    f.write('</edges>\n')

# ----------------------------
# Define 9 Entry Nodes (g0 ~ g8)
# ----------------------------
entry_nodes = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),  # n01~n05
    (1, 6), (2, 6), (3, 6), (4, 6), (5, 6),  # n16~n56
    (6, 1), (6, 2), (6, 3), (6, 4), (6, 5),  # n61~n65
    (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),  # n10~n50
]
exit_nodes = entry_nodes.copy()

# ----------------------------
# Generate Routes (Manhattan path)
# ----------------------------
def manhattan_path(x1, y1, x2, y2):
    path = []
    x, y = x1, y1

    while (x, y) != (x2, y2):
        options = []
        if x != x2:
            options.append('x')
        if y != y2:
            options.append('y')

        if not options:
            break  # should not happen

        move = random.choice(options)
        if move == 'x':
            next_x = x + 1 if x2 > x else x - 1
            path.append(f"n_{x}_{y}_n_{next_x}_{y}")
            x = next_x
        else:  # move == 'y'
            next_y = y + 1 if y2 > y else y - 1
            path.append(f"n_{x}_{y}_n_{x}_{next_y}")
            y = next_y

    return path



with open(ROUTES_FILE, 'w') as f:
    f.write('<routes>\n')

    vehicle_id = 0
    step = 0.0
    for _ in range(18000):
        (x1, y1) = random.choice(entry_nodes)
        (x2, y2) = random.choice(exit_nodes)
        while (x1, y1) == (x2, y2):
            (x2, y2) = random.choice(entry_nodes)

        start_id = f"n_{x1}_{y1}"
        end_id = f"n_{x2}_{y2}"
        route_id = f"r{vehicle_id}"
        edges = manhattan_path(x1, y1, x2, y2)
        route_edges = ' '.join(edges)

        speed = round(random.uniform(5.0, 15.0), 1)  # 每台車的 maxSpeed
        vtype_id = f"car{vehicle_id}"

        f.write(f'    <vType id="{vtype_id}" accel="2.6" decel="4.5" maxSpeed="{speed}" length="5.0"/>\n')
        f.write(f'    <route id="{route_id}" edges="{route_edges}"/>\n')
        f.write(f'    <vehicle id="veh{vehicle_id}" type="{vtype_id}" route="{route_id}" depart="{step}"/>\n')

        vehicle_id += 1
        step += random.uniform(0, 4)

    f.write('</routes>\n')



# ----------------------------
# Generate Net file using netconvert
# ----------------------------
subprocess.run([
    'netconvert',
    '--node-files', NODES_FILE,
    '--edge-files', EDGES_FILE,
    '--output-file', NET_FILE,
])

# ----------------------------
# Generate SUMO Config file
# ----------------------------
with open(SUMOCFG_FILE, 'w') as f:
    f.write('<configuration>\n')
    f.write('    <input>\n')
    f.write(f'        <net-file value="{NET_FILE}"/>\n')
    f.write(f'        <route-files value="{ROUTES_FILE}"/>\n')
    f.write('    </input>\n')
    f.write('    <time>\n')
    f.write('        <begin value="0"/>\n')
    f.write('        <end value="10000"/>\n')
    f.write('        <step-length value="1.0"/>\n')
    f.write('    </time>\n')
    f.write('    <processing>\n')
    f.write('        <collision.action value="none"/>\n')
    f.write('        <ignore-route-errors value="true"/>\n')
    f.write('    </processing>\n')
    f.write('</configuration>\n')

print("finish")
