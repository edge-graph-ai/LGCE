import torch
import os, re

class GraphData:
    """
    一个封装数据图结构的类，包含：
    - num_vertices: 节点数
    - num_edges: 边数
    - num_labels: 标签种类数
    - vertices: {
        vid: {
            'label': vlabel,
            'degree': vdegree
        }
      }
    - edges: [(src, dst), ...]
    - edge_index: [ [src1, src2, ...], [dst1, dst2, ...] ]  # 2 x num_edges
    """

    def __init__(self):
        self.num_vertices = 0
        self.num_edges = 0
        self.num_labels = 0  # 初始化标签种类数
        self.vertices = {}
        self.edges = []
        self.edge_index = [[], []]  # 2行(起点行, 终点行)

    def __repr__(self):
        """让打印对象时更直观"""
        return (f"GraphData(num_vertices={self.num_vertices}, "
                f"num_edges={self.num_edges}, "
                f"num_labels={self.num_labels})")

    def get_vertex_info(self, vid):
        """返回某个顶点的所有信息字典"""
        return self.vertices.get(vid, None)

    def add_vertex(self, vid, label,degree):
        self.vertices[vid] = {
            'label': label,
            'degree': degree
        }


    def add_edge(self, src, dst):
        """添加一条 (src, dst) 边，同时更新顶点的出/入邻居列表"""
        self.edges.append((src, dst))
        self.edge_index[0].append(src)
        self.edge_index[1].append(dst)

    def cal_num_labels(self):
        vertex_values = self.vertices.values()
        unique_labels = set(v['label'] for v in vertex_values)
        self.num_labels = len(unique_labels)
def load_graph_from_file(filename):
    """
    从特定格式的文本文件中读取图数据，并返回一个 GraphData 实例。

    文件格式：
    1) 第一行: "t n1 n2"
       - t: 固定符号(可忽略)
       - n1: 节点数
       - n2: 边数
    2) 接下来 n1 行，每行: "v vid vlabel vdegree"
       - v: 固定符号(可忽略)
       - vid: 顶点ID (int)
       - vlabel: 顶点label (string / int)
       - vdegree: 顶点度数 (int)
    3) 接下来 n2 行，每行: "n ns nd"
       - n: 固定符号(可忽略)
       - ns: 边起点ID
       - nd: 边终点ID
    """
    graph = GraphData()

    with open(filename, 'r') as f:
        # 1. 读取第一行: t n1 n2
        first_line = f.readline().strip()
        items = first_line.split()
        # t_symbol = items[0]  # 't'，可忽略或检查
        n1 = int(items[1])  # 顶点数
        n2 = int(items[2])  # 边数

        # 更新 graph 的 meta 信息
        graph.num_vertices = n1
        graph.num_edges = n2

        # 2. 读取 n1 行 (节点信息)
        for _ in range(n1):
            line = f.readline().strip()
            v_items = line.split()
            # v_symbol = v_items[0]  # 'v'
            vid = int(v_items[1])
            vlabel = v_items[2]
            vdegree = int(v_items[3])

            # 使用类的方法添加顶点
            graph.add_vertex(vid, vlabel,vdegree)

        # 3. 读取 n2 行 (边信息)
        for _ in range(n2):
            line = f.readline().strip()
            e_items = line.split()
            # n_symbol = e_items[0]  # 'n'
            ns = int(e_items[1])
            nd = int(e_items[2])

            # 使用类的方法添加边
            graph.add_edge(ns, nd)
    graph.cal_num_labels()
    return graph

def load_true_cardinalities_from_file(file_path):
    """
    从文件中提取每行包含 #Matches: 的匹配数，并返回一个列表。

    参数:
        file_path (str): 文件的路径。

    返回:
        list[int]: 包含所有提取的 #Matches: 数据的列表。
    """
    true_cardinalities = []

    # 打开并读取文件
    with open(file_path, 'r') as file:
        for line in file:
            # 检查行是否包含匹配的关键词
            if "#Matches:" in line:
                # 提取 #Matches: 后的数据
                parts = line.split("#Matches:")
                if len(parts) > 1:
                    try:
                        # 提取并转换为整数
                        match_count = int(parts[1].strip())
                        # 将结果追加到列表中
                        true_cardinalities.append(match_count)
                    except ValueError:
                        # 如果无法转换为整数，则跳过该行
                        print(f"Warning: Unable to parse matches in line: {line.strip()}")

    return true_cardinalities


def load_true_cardinalities_aligned(matches_path, query_ids):
    """
    返回与 query_ids 一一对应的 true_cardinalities。
    优先解析：
      1) 'Query Graph: <path/to/xxx.graph> ... #Matches: N'
      2) 任意位置出现 'xxx.graph ... #Matches: N'
    最后兜底：只出现 '#Matches: N' 时按行顺序对齐（要求数量一致）。
    """
    name_to_cnt = {}
    counts_seq = []

    # ① 专门匹配 'Query Graph: <路径>.graph' 的格式（最可靠）
    pat_query_graph = re.compile(
        r'Query\s*Graph:\s*(?P<qpath>.+?\.graph)\b.*?#Matches:\s*(?P<count>\d+)',
        re.IGNORECASE
    )
    # ② 兜底：行内任意位置出现 '<名字>.graph ... #Matches: N'
    pat_any_graph = re.compile(
        r'(?P<name>[\w\-.]+\.graph).*?#Matches:\s*(?P<count>\d+)',
        re.IGNORECASE
    )
    # ③ 仅有 '#Matches: N'
    pat_count_only = re.compile(r'#Matches:\s*(?P<count>\d+)', re.IGNORECASE)

    with open(matches_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            m1 = pat_query_graph.search(s)
            if m1:
                qpath = m1.group('qpath')
                count = int(m1.group('count'))
                base = os.path.splitext(os.path.basename(qpath))[0]
                name_to_cnt[base] = count
                continue

            m2 = pat_any_graph.search(s)
            if m2:
                name = os.path.splitext(os.path.basename(m2.group('name')))[0]
                count = int(m2.group('count'))
                name_to_cnt[name] = count
                continue

            m3 = pat_count_only.search(s)
            if m3:
                counts_seq.append(int(m3.group('count')))
                continue

    # 按名称对齐（首选）
    if name_to_cnt:
        missing = [qid for qid in query_ids if qid not in name_to_cnt]
        extra   = [k for k in name_to_cnt.keys() if k not in set(query_ids)]
        if missing:
            # 打印前几项帮助定位
            raise ValueError(
                f"在 {matches_path} 未找到这些查询的 #Matches：{missing[:5]}"
                + (" ..." if len(missing) > 5 else "")
            )
        if extra:
            print(f"[Warn] {matches_path} 有额外项（已忽略）：{extra[:5]}" + (" ..." if len(extra) > 5 else ""))
        return [name_to_cnt[qid] for qid in query_ids]

    # 只好按顺序对齐
    if len(counts_seq) != len(query_ids):
        raise ValueError(
            f"{matches_path} 仅包含顺序数值且数量不等：counts={len(counts_seq)} vs queries={len(query_ids)}。\n"
            f"请确保文件里包含查询名，或保证两者顺序完全一致。"
        )
    print("[Warn] 基数文件没有包含查询名称，按行顺序对齐 —— 请确保其生成顺序与按文件名排序后的查询一致！")
    return counts_seq