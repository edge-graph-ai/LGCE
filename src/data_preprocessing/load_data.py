import torch
import os, re

class GraphData:

    def __init__(self):
        self.num_vertices = 0
        self.num_edges = 0
        self.num_labels = 0  
        self.vertices = {}
        self.edges = []
        self.edge_index = [[], []]  

    def __repr__(self):
        return (f"GraphData(num_vertices={self.num_vertices}, "
                f"num_edges={self.num_edges}, "
                f"num_labels={self.num_labels})")

    def get_vertex_info(self, vid):
        return self.vertices.get(vid, None)

    def add_vertex(self, vid, label,degree):
        self.vertices[vid] = {
            'label': label,
            'degree': degree
        }


    def add_edge(self, src, dst):
        self.edges.append((src, dst))
        self.edge_index[0].append(src)
        self.edge_index[1].append(dst)

    def cal_num_labels(self):
        vertex_values = self.vertices.values()
        unique_labels = set(v['label'] for v in vertex_values)
        self.num_labels = len(unique_labels)
def load_graph_from_file(filename):
    
    graph = GraphData()

    with open(filename, 'r') as f:
       
        first_line = f.readline().strip()
        items = first_line.split()
        
        n1 = int(items[1])  
        n2 = int(items[2])  


        graph.num_vertices = n1
        graph.num_edges = n2


        for _ in range(n1):
            line = f.readline().strip()
            v_items = line.split()
            vid = int(v_items[1])
            vlabel = v_items[2]
            vdegree = int(v_items[3])

            
            graph.add_vertex(vid, vlabel,vdegree)


        for _ in range(n2):
            line = f.readline().strip()
            e_items = line.split()

            ns = int(e_items[1])
            nd = int(e_items[2])


            graph.add_edge(ns, nd)
    graph.cal_num_labels()
    return graph

def load_true_cardinalities_from_file(file_path):
    
    true_cardinalities = []


    with open(file_path, 'r') as file:
        for line in file:

            if "#Matches:" in line:

                parts = line.split("#Matches:")
                if len(parts) > 1:
                    try:
                        match_count = int(parts[1].strip())

                        true_cardinalities.append(match_count)
                    except ValueError:
                        print(f"Warning: Unable to parse matches in line: {line.strip()}")

    return true_cardinalities


def load_true_cardinalities_aligned(matches_path, query_ids):
    
    name_to_cnt = {}
    counts_seq = []


    pat_query_graph = re.compile(
        r'Query\s*Graph:\s*(?P<qpath>.+?\.graph)\b.*?#Matches:\s*(?P<count>\d+)',
        re.IGNORECASE
    )
   
    pat_any_graph = re.compile(
        r'(?P<name>[\w\-.]+\.graph).*?#Matches:\s*(?P<count>\d+)',
        re.IGNORECASE
    )
    
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

    
    if name_to_cnt:
        missing = [qid for qid in query_ids if qid not in name_to_cnt]
        extra   = [k for k in name_to_cnt.keys() if k not in set(query_ids)]
        if missing:
            raise ValueError(
                f"在 {matches_path} 未找到这些查询的 #Matches：{missing[:5]}"
                + (" ..." if len(missing) > 5 else "")
            )
        if extra:
            print(f"[Warn] {matches_path} 有额外项（已忽略）：{extra[:5]}" + (" ..." if len(extra) > 5 else ""))
        return [name_to_cnt[qid] for qid in query_ids]

    if len(counts_seq) != len(query_ids):
        raise ValueError(
            f"{matches_path} 仅包含顺序数值且数量不等：counts={len(counts_seq)} vs queries={len(query_ids)}。\n"
            f"请确保文件里包含查询名，或保证两者顺序完全一致。"
        )
    print("[Warn] 基数文件没有包含查询名称，按行顺序对齐 —— 请确保其生成顺序与按文件名排序后的查询一致！")
    return counts_seq