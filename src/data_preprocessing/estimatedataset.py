# estimatedataset.py
import torch
from torch.utils.data import Dataset

class EstimateDataset(Dataset):
    """
    一个轻量的数据集：为每个查询图配对：
      - 同一份数据子图列表（PyG Data）
      - 它的真实基数
      - 它的 query_id（例如 'query_dense_12_165'），便于做分层/过滤

    Args:
        data_pygs: List[List[PyG Data]]  —— 数据图的子图列表；通常所有样本共享同一份
        query_pygs: List[PyG Data]       —— 每个查询图各一个
        true_cardinalities: List[int]    —— 与 query_pygs 对齐的一一对应标签
        query_ids: Optional[List[str]]   —— 与 query_pygs 同长；若不给就用 q_{i}
    """
    def __init__(self, data_pygs, query_pygs, true_cardinalities, query_ids=None):
        assert len(query_pygs) == len(true_cardinalities), \
            f"query_pygs={len(query_pygs)} vs labels={len(true_cardinalities)}"
        if query_ids is not None:
            assert len(query_ids) == len(query_pygs), "query_ids length mismatch"
        self.data_pygs = data_pygs
        self.query_pygs = query_pygs
        self.true_cardinalities = true_cardinalities
        self.query_ids = query_ids if query_ids is not None else [f"q_{i}" for i in range(len(query_pygs))]

    def __len__(self):
        return len(self.query_pygs)

    def __getitem__(self, idx):
        q = self.query_pygs[idx]
        y = int(self.true_cardinalities[idx])
        sample = ({
            'data_graph': self.data_pygs,     # 共享一份子图列表
            'query_graph': q,
            'query_id': self.query_ids[idx],
        }, torch.tensor(y, dtype=torch.long))
        return sample
