# 需要将数据图分割为固定数量K的子图
# 划分的方法大致如下
# 若数据图共有N个顶点，首先分别按顺序划分为K个N/K的顶点集。
# 然后根据查询图的模式，分别对K个顶点集进行筛选，过滤出不可能在查询结果中出现的顶点。
# 最后将K个顶点集的导出子图作为数据图的K个子图。
# 数据图的格式如下
# t 5 5
# v 0 0 2
# v 1 1 2
# v 2 0 3
# v 3 1 2
# v 4 2 1
# e 0 1 0
# e 0 2 0
# e 2 3 0
# e 1 3 0
# e 2 4 0
# 分解子图的思路，结合后续的任务，我仔细思考了一下，最值得做的一个剪枝就是根据查询图的节点label进行剪枝
# 下面我先加入这个剪枝策略，实现一些代码
from collections import Counter, defaultdict, deque
import multiprocessing
import random
import numpy as np
from src.data_preprocessing.load_data import GraphData
from collections import deque, defaultdict


def prune_graph(graph_data):
    """
    根据提供的 GraphData 图结构，执行图剪枝操作，返回剪枝后的新 GraphData 实例。
    剪枝策略：
      1. 基于度数阈值移除度数过低或过高的节点；
      2. 基于 PageRank 节点重要性移除影响较小的节点；
      3. 基于生成树保留图的骨架结构（边稀疏化）。
    剪枝后会更新 vertices、edges、edge_index 以及 num_vertices 和 num_edges 信息。
    """
    # 提取原图数据
    vertices = graph_data.vertices  # 原始节点字典 {node_id: {'label': ..., 'degree': ...}, ...}
    edges = graph_data.edges  # 原始边列表 [(u, v), (u2, v2), ...]
    edge_index = graph_data.edge_index  # 原始边索引结构（起点和终点的对应关系）

    # 1. 基于度数阈值的节点剪枝

    # 设置度数阈值：这里我们示例性地将度数<=1视为过低，度数过高可以根据图规模动态决定。
    low_degree_thresh = 1
    # 高度数阈值设为超过节点总数的一定比例，此处设为50%（即连接超过一半节点的超级节点）作为示例。
    high_degree_thresh = 0.5 * len(vertices) if len(vertices) > 0 else 0

    # 找出需要移除的节点：度数小于等于低阈值，或度数大于等于高阈值
    nodes_to_remove = set()
    for node_id, attr in vertices.items():
        deg = attr.get('degree', 0)
        if deg <= low_degree_thresh or deg >= high_degree_thresh:
            nodes_to_remove.add(node_id)
    # 从原节点集中移除这些节点
    remaining_vertices = {nid: data for nid, data in vertices.items() if nid not in nodes_to_remove}

    # 从边列表中移除涉及上述节点的边
    remaining_edges = []
    for (u, v) in edges:
        if u in nodes_to_remove or v in nodes_to_remove:
            # 如果边的任一端点被移除了，则跳过该边（不保留）
            continue
        remaining_edges.append((u, v))

    # 更新剩余图的邻接信息，以便后续计算（构建邻接表）
    adj_list = {}  # 用于存储剩余节点的邻居列表
    for (u, v) in remaining_edges:
        # 对于无向图，我们将邻接关系记录两次；如果是有向图，则仅记录单向邻接。
        if u not in adj_list:
            adj_list[u] = []
        if v not in adj_list:
            adj_list[v] = []
        adj_list[u].append(v)
        adj_list[v].append(u)
    # 确保所有剩余节点都在邻接表中（即使某些节点可能暂时没有边）
    for nid in remaining_vertices:
        adj_list.setdefault(nid, [])

    # 2. 基于节点重要性的剪枝（使用 PageRank 作为重要性指标）

    # PageRank参数
    damping = 0.85  # 阻尼系数
    max_iterations = 20
    min_delta = 1e-6  # 收敛阈值

    num_nodes = len(remaining_vertices)
    if num_nodes > 0:
        # 初始化每个节点的 PageRank 值
        pr_values = {nid: 1.0 / num_nodes for nid in remaining_vertices}
        # 预计算每个节点的出度（用于PageRank公式中的归一化）；这里使用无向图邻居数近似出度
        out_degree = {nid: len(adj_list[nid]) for nid in remaining_vertices}

        for _ in range(max_iterations):
            diff = 0  # 记录本轮迭代值变化用于判断收敛
            new_pr = {}
            for nid in remaining_vertices:
                # 计算来自所有入邻居（在无向图中邻居即入邻居）的PageRank贡献之和
                # PR公式: new_pr[nid] = (1-damping)/N + damping * sum(pr[nb]/deg(nb) for nb in neighbors_of_nid)
                rank_sum = 0.0
                for nb in adj_list[nid]:
                    # 如果邻居节点有出度，则贡献其PR值的1/out_degree
                    if out_degree.get(nb, 0) > 0:
                        rank_sum += pr_values[nb] / out_degree[nb]
                new_pr_val = (1 - damping) / num_nodes + damping * rank_sum
                new_pr[nid] = new_pr_val
                diff += abs(new_pr_val - pr_values[nid])
            pr_values = new_pr
            # 如果PageRank值变化小于阈值，则提早停止迭代（收敛）
            if diff < min_delta:
                break

        # 计算PageRank的平均值，用于设定重要性阈值
        avg_pr = sum(pr_values.values()) / num_nodes
        pr_threshold = 0.5 * avg_pr  # 示例阈值：低于平均值一半的节点视为不重要

        # 找出PageRank低于阈值的节点
        pruned_by_pr = {nid for nid, pr in pr_values.items() if pr < pr_threshold}

        # 从剩余节点和边中移除这些PageRank较低的节点
        for nid in pruned_by_pr:
            remaining_vertices.pop(nid, None)
            # 对邻接表的清理：虽然下一步会重建边集和邻接关系，这里可选地清理
            if nid in adj_list:
                # 对于无向图，也需从其他邻居列表中移除该节点
                for nb in adj_list[nid]:
                    if nb in adj_list:
                        adj_list[nb] = [x for x in adj_list[nb] if x != nid]
                adj_list.pop(nid, None)
        # 更新边列表，移除包含被删节点的边
        remaining_edges = [(u, v) for (u, v) in remaining_edges if u not in pruned_by_pr and v not in pruned_by_pr]
        # 更新剩余节点数
        num_nodes = len(remaining_vertices)
    else:
        pr_values = {}

    # 3. 基于边稀疏化的剪枝（提取图的骨架生成树）

    # 为了保留主要结构，我们在每个连通分量上构造一个生成树，仅保留生成树中的边。
    new_edges = []
    visited = set()
    # 对每个尚未访问的节点，执行一次深度优先搜索/广度优先搜索来获取生成树
    for start_node in list(remaining_vertices.keys()):
        if start_node in visited:
            continue
        # 初始化队列进行 BFS（广度优先搜索）以构建生成树
        queue = [start_node]
        visited.add(start_node)
        while queue:
            u = queue.pop(0)
            # 遍历u的邻居（注意：此时邻居关系基于remaining_edges更新后的adj_list）
            for v in adj_list.get(u, []):
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
                    # u -> v 这条边用于生成树骨架，保留下来
                    new_edges.append((u, v))
    # 注：上述过程确保每个连通分量得到一棵生成树。如果图是无向的，生成树包含 |comp| - 1 条边。

    # 4. 构建剪枝后的 GraphData

    # 重新计算每个保留节点的新度数（根据 new_edges）
    new_adj_list = {nid: [] for nid in remaining_vertices}  # 重建邻接表以计算度数
    for (u, v) in new_edges:
        new_adj_list[u].append(v)
        new_adj_list[v].append(u)
    # 更新 vertices 字典，保留原 label，更新 degree
    pruned_vertices = {}
    for nid, data in remaining_vertices.items():
        new_degree = len(new_adj_list.get(nid, []))
        pruned_vertices[nid] = {
            'label': data.get('label', None),
            'degree': new_degree
        }
    # 更新 edge_index 结构。假设edge_index为二维列表形式：[sources_list, targets_list]
    pruned_edge_index = [[], []]
    for (u, v) in new_edges:
        pruned_edge_index[0].append(u)
        pruned_edge_index[1].append(v)
    # 如果是无向图，我们不重复添加反向边；如果图为有向图，此处应根据需求调整 edge_index 构建方式。

    # 生成新的 GraphData 对象并返回
    pruned_graph = GraphData()
    pruned_graph.vertices = pruned_vertices
    pruned_graph.edges = new_edges
    pruned_graph.edge_index = pruned_edge_index
    pruned_graph.num_vertices = len(pruned_vertices)
    pruned_graph.num_edges = len(new_edges)
    return pruned_graph
def prune_data_graph_by_query(data_graph, query_graph):
    """
    改进版剪枝：不仅利用查询图中各标签的最大度数要求作为初筛条件，
    还利用候选顶点诱导子图中的实际邻接关系，迭代去除那些在局部图中不满足度数要求的顶点，
    从而更充分地剪掉无用部分，降低后续计算和内存消耗。

    参数：
      data_graph: GraphData 类的实例（原数据图），其 vertices 中应包含 'label' 和全局 'degree'
      query_graph: GraphData 类的实例（查询图），其 vertices 中应包含 'label' 和 'degree'
    返回：
      pruned_graph: GraphData 类的实例（剪枝后的图）
    """

    # 1. 从查询图中获取所有标签；具体标签为除去通配符 "-1"
    query_labels = {vinfo['label'] for vinfo in query_graph.vertices.values()}
    specific_labels = query_labels - {"-1"}
    wildcard_present = ("-1" in query_labels)

    # 2. 对于每个具体标签，计算查询图中该标签顶点的最大度数要求
    required_degree = {}
    for vinfo in query_graph.vertices.values():
        label = vinfo['label']
        if label in specific_labels:
            deg = int(vinfo.get('degree', 0))
            required_degree[label] = max(required_degree.get(label, 0), deg)

    # 3. 初始候选集合：遍历数据图中的顶点
    # 对于标签在具体标签中的顶点，要求其全局度数 >= 要求度数；
    # 对于其他顶点，如果查询图存在通配符，则保留；否则剔除。
    candidates = {}
    for dvid, dvinfo in data_graph.vertices.items():
        label = dvinfo['label']
        ddeg = int(dvinfo.get('degree', 0))
        if label in specific_labels:
            if ddeg >= required_degree.get(label, 0):
                candidates[dvid] = dvinfo.copy()  # 复制字典
        else:
            if wildcard_present:
                candidates[dvid] = dvinfo.copy()

    # 4. 迭代剪枝：利用候选顶点诱导子图中的实际连边情况进一步筛除不满足度数要求的顶点
    changed = True
    while changed:
        changed = False
        removal_list = []

        # 计算诱导子图中，每个候选顶点的实际度数
        induced_degrees = {}
        for src, dst in data_graph.edges:
            if src in candidates and dst in candidates:
                induced_degrees[src] = induced_degrees.get(src, 0) + 1
                induced_degrees[dst] = induced_degrees.get(dst, 0) + 1

        # 对于每个具体标签的候选顶点，若其在诱导子图中的实际度数低于查询要求，则标记为删除
        for dvid, dvinfo in list(candidates.items()):
            label = dvinfo['label']
            if label in specific_labels:
                induced_deg = induced_degrees.get(dvid, 0)
                if induced_deg < required_degree.get(label, 0):
                    removal_list.append(dvid)

        if removal_list:
            for vid in removal_list:
                del candidates[vid]
            changed = True  # 有变化则继续下一轮迭代

    # 5. 重新构建剪枝后的边集合：仅保留两端均在候选集合内的边
    pruned_edges = []
    for src, dst in data_graph.edges:
        if src in candidates and dst in candidates:
            pruned_edges.append((src, dst))

    # 6. 构造剪枝后的图对象
    pruned_graph = GraphData()
    pruned_graph.vertices = candidates
    pruned_graph.edges = pruned_edges
    pruned_graph.num_vertices = len(candidates)
    pruned_graph.num_edges = len(pruned_edges)

    return pruned_graph


# def prune_data_graph_by_query(data_graph, query_graph):
#     """
#     根据 query_graph 的顶点 label 对 data_graph 进行剪枝：
#       - 如果查询图中存在 label 为 -1 的顶点，则认为该查询条件可以匹配所有数据图顶点，
#         因此不对数据图顶点进行 label 过滤（即尽可能保留更多的顶点）。
#       - 否则，只保留数据图中 label 在查询图中出现的顶点，
#         同时移除不相关的顶点和与之关联的边，
#         并更新剩余顶点的出/入邻居列表以及度数。
#     参数：
#       data_graph: GraphData 类的实例 (原数据图)
#       query_graph: GraphData 类的实例 (查询图)
#     返回：
#       pruned_graph: GraphData 类的实例 (剪枝后数据图)
#     """
#
#     # 1. 获取查询图中的所有 label
#     query_labels = set()
#     for qvid, qvinfo in query_graph.vertices.items():
#         query_labels.add(qvinfo['label'])
#
#     # 判断是否包含通配符 -1
#     wildcard = ("-1" in query_labels)
#
#     # 2. 构造一个新的 GraphData 用于存放“剪枝后”的结果
#     pruned_graph = GraphData()
#
#     # 准备：记录哪些节点会被保留
#     keep_vertices = {}
#
#     # 3. 遍历 data_graph 中的顶点，根据策略选择保留哪些顶点
#     for dvid, dvinfo in data_graph.vertices.items():
#         # 如果查询图包含 -1（通配符），则所有数据图顶点都保留；
#         # 否则只保留 label 出现在 query_labels 中的顶点
#         if wildcard or dvinfo['label'] in query_labels:
#             keep_vertices[dvid] = {
#                 'label': dvinfo['label'],
#                 # 如有其他属性也可以根据需要添加
#             }
#
#     # 4. 遍历 data_graph 的边，只保留两端均在 keep_vertices 中的边
#     new_edges = []
#     src_list = []
#     dst_list = []
#     for (src, dst) in data_graph.edges:
#         if src in keep_vertices and dst in keep_vertices:
#             new_edges.append((src, dst))
#             src_list.append(src)
#             dst_list.append(dst)
#
#     # 5. 将保留后的顶点与边信息放进 pruned_graph
#     pruned_graph.vertices = keep_vertices
#     pruned_graph.edges = new_edges
#     pruned_graph.edge_index = [src_list, dst_list]
#
#     # 6. 更新图的顶点数和边数
#     pruned_graph.num_vertices = len(pruned_graph.vertices)
#     pruned_graph.num_edges = len(pruned_graph.edges)
#
#     return pruned_graph


def find_min_label_vertex_count(pruned_data_graph):
    """
    根据 label 对顶点分类，找到个数最少的 label，并返回该 label 对应的顶点个数。

    参数：
      pruned_data_graph: GraphData 类的实例（剪枝后的数据图）

    返回：
      最少顶点个数
    """
    # 1. 收集每个顶点的标签
    label_counter = Counter()

    for vinfo in pruned_data_graph.vertices.values():
        label_counter[vinfo['label']] += 1

    # 2. 找到最少的标签个数
    min_count = min(label_counter.values())  # 获取最少的顶点个数

    return min_count
def partition_graph_randomly(data_graph, K):
    """
    将 data_graph (GraphData 实例) 随机划分成 K 个子图 (GraphData)。

    步骤：
      1. 收集所有节点ID，并随机打乱顺序。
      2. 均匀分配到 K 个子图中。
      3. 对每个子图，保留其内部的边，并重新映射节点索引。
      4. 更新子图中每个节点的出/入邻居列表及度数。

    参数：
      data_graph: GraphData 类的实例 (剪枝后的数据图)。
      K: 要划分的子图数量。

    返回：
      subgraphs: List[GraphData]，长度为 K，每个元素是一个子图。
    """
    import numpy as np

    # 1. 收集所有节点ID，并随机打乱
    all_nodes = list(data_graph.vertices.keys())
    all_nodes = np.array(all_nodes)
    np.random.shuffle(all_nodes)
    all_nodes = all_nodes.tolist()

    # 2. 平分到K份 (若不能整除，也尽量均匀分配)
    N = len(all_nodes)
    chunk_size = (N + K - 1) // K  # 向上取整
    partitions = []
    start_idx = 0
    for _ in range(K):
        end_idx = min(start_idx + chunk_size, N)
        node_subset = all_nodes[start_idx:end_idx]
        partitions.append(node_subset)
        start_idx = end_idx

    # partitions 现在是一个长度为K的列表，每个元素是该分区的节点列表

    # 3. 为每个分区构造子图
    subgraphs = []
    for i in range(K):
        node_ids_in_part = set(partitions[i])

        # 构造一个新的GraphData
        subg = GraphData()

        # 创建节点ID映射，从原始ID映射到子图中的新ID
        id_map = {old_id: new_id for new_id, old_id in enumerate(node_ids_in_part)}

        # 先收集本分区的顶点信息
        for old_id in node_ids_in_part:
            vlabel = data_graph.vertices[old_id]['label']
            # 暂时将出/入邻居留空，degree=0，后续再填
            subg.vertices[id_map[old_id]] = {
                'label': vlabel,
                'original_id': old_id  # 添加原始数据图的顶点ID
            }

        # 4. 扫描 data_graph 中所有边，把属于这个分区内部的边保留并重新映射
        src_list, dst_list = [], []
        edges_temp = []
        for (src, dst) in data_graph.edges:
            if src in node_ids_in_part and dst in node_ids_in_part:
                # 边在分区内部，保留，并重新映射索引
                new_src = id_map[src]
                new_dst = id_map[dst]
                edges_temp.append((new_src, new_dst))
                src_list.append(new_src)
                dst_list.append(new_dst)

        subg.edges = edges_temp
        subg.edge_index = [src_list, dst_list]
        subg.num_edges = len(edges_temp)
        subg.num_vertices = len(subg.vertices)
        subgraphs.append(subg)

    return subgraphs

def partition_graph_bfs(data_graph, K):
    """
    用 BFS 生长的方式，将 data_graph 划分成 K 个子图 (GraphData)。

    步骤：
      1. 随机选 K 个 seed 节点；
      2. 对每个 seed 做 BFS 生长，逐步把未分配的邻居加入对应子图，直到子图规模接近 ceil(N/K)；
      3. 把剩余还未分配的节点随机分配到规模未满的子图中；
      4. 对每个子图，保留内部边并重新映射节点索引。

    返回：
      subgraphs: List[GraphData]，长度为 K，每个元素是一个子图。
    """
    # 1. 构建邻接表
    adj = defaultdict(list)
    for u, v in data_graph.edges:
        adj[u].append(v)
        adj[v].append(u)

    all_nodes = list(data_graph.vertices.keys())
    N = len(all_nodes)
    target_size = (N + K - 1) // K  # ceil(N/K)

    # 2. 随机选种子
    seeds = random.sample(all_nodes, K)
    assigned = set(seeds)

    # 3. 初始化每个 partition 的队列和成员列表
    queues    = {s: deque([s]) for s in seeds}
    partitions= {s: [s]        for s in seeds}

    # 4. BFS 生长
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
                    # 一旦填满就跳出
                    if len(part) >= target_size:
                        break
            # 如果当前 u 的所有邻居都已分配，则继续下一个 u

    # 5. 剩余节点随机分配到还没满的子图
    rest = [n for n in all_nodes if n not in assigned]
    random.shuffle(rest)
    # 按轮询方式分给规模 < target_size 的分区
    underfull = [s for s in seeds if len(partitions[s]) < target_size]
    idx = 0
    for n in rest:
        if not underfull:
            break
        s = underfull[idx % len(underfull)]
        partitions[s].append(n)
        assigned.add(n)
        idx += 1
        # 如果当前分区达标则从 underfull 中移除
        if len(partitions[s]) >= target_size:
            underfull.remove(s)

    # 6. 构造 GraphData 子图列表
    subgraphs = []
    for s in seeds:
        node_ids = partitions[s]
        id_map   = {old: new for new, old in enumerate(node_ids)}
        subg     = GraphData()

        # 顶点
        for old in node_ids:
            vinfo = data_graph.vertices[old]
            subg.vertices[id_map[old]] = {
                'label':       vinfo['label'],
                'degree':      vinfo['degree'],
                'original_id': old
            }

        # 边
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
