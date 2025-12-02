# estimatedataset.py
import torch
from torch.utils.data import Dataset

class EstimateDataset(Dataset):
    """


    Args:
        data_pygs: List[List[PyG Data]]   
        query_pygs: List[PyG Data]        
        true_cardinalities: List[int]    
        query_ids: Optional[List[str]]   
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
            'data_graph': self.data_pygs,     
            'query_graph': q,
            'query_id': self.query_ids[idx],
        }, torch.tensor(y, dtype=torch.long))
        return sample
