
from collections import Counter, defaultdict, deque
import multiprocessing
import random
import numpy as np
from src.data_preprocessing.load_data import GraphData
from collections import deque, defaultdict

def partition_graph_bfs(data_graph, K):

    
    adj = defaultdict(list)
    for u, v in data_graph.edges:
        adj[u].append(v)
        adj[v].append(u)

    all_nodes = list(data_graph.vertices.keys())
    N = len(all_nodes)
    target_size = (N + K - 1) // K  # ceil(N/K)

    
    seeds = random.sample(all_nodes, K)
    assigned = set(seeds)

    
    queues    = {s: deque([s]) for s in seeds}
    partitions= {s: [s]        for s in seeds}

   
    for s in seeds:
        q = queues[s]
        part = partitions[s]
        while q and len(part) < target_size:
            u = q.popleft()
            for nbr in adj[u]:
                if nbr not in assigned:
                    assigned.add(nbr)
                    part.append(nbr)
                    q.append(nbr)
                 
                    if len(part) >= target_size:
                        break
            

  
    rest = [n for n in all_nodes if n not in assigned]
    random.shuffle(rest)

    underfull = [s for s in seeds if len(partitions[s]) < target_size]
    idx = 0
    for n in rest:
        if not underfull:
            break
        s = underfull[idx % len(underfull)]
        partitions[s].append(n)
        assigned.add(n)
        idx += 1

        if len(partitions[s]) >= target_size:
            underfull.remove(s)


    subgraphs = []
    for s in seeds:
        node_ids = partitions[s]
        id_map   = {old: new for new, old in enumerate(node_ids)}
        subg     = GraphData()

       
        for old in node_ids:
            vinfo = data_graph.vertices[old]
            subg.vertices[id_map[old]] = {
                'label':       vinfo['label'],
                'degree':      vinfo['degree'],
                'original_id': old
            }


        src_list, dst_list = [], []
        for u, v in data_graph.edges:
            if u in id_map and v in id_map:
                nu, nv = id_map[u], id_map[v]
                src_list.append(nu)
                dst_list.append(nv)
        subg.edge_index  = [src_list, dst_list]
        subg.edges       = list(zip(src_list, dst_list))
        subg.num_vertices= len(node_ids)
        subg.num_edges   = len(src_list)

        subgraphs.append(subg)

    return subgraphs
