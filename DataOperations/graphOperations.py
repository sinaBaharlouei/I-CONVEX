import networkx
from queue import Queue


def find_connected_components(G):
    cluster_dict = {}
    marked_dict = {}

    queue = Queue()
    nodes = G.nodes()

    for node in nodes:
        marked_dict[node] = False
        cluster_dict[node] = -1

    cluster_index = 0

    for node in nodes:

        if marked_dict[node]:
            continue

        else:
            queue.put(node)

        while not queue.empty():

            current_node = queue.get()
            if cluster_dict[current_node] < 0:
                cluster_index += 1
                cluster_dict[current_node] = cluster_index
                marked_dict[current_node] = True

            edges = G.edges(current_node)
            for edge in edges:
                neighbor = edge[1]
                if not marked_dict[neighbor]:
                    cluster_dict[neighbor] = cluster_dict[current_node]
                    queue.put(neighbor)
                    marked_dict[neighbor] = True

    return cluster_dict
