import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from typing import Optional, Sequence

def custom_collate_fn(batch):
    """
    
      1) ({'data_graph': List[PyG Data], 'query_graph': PyG Data}, y)
      2) (List[PyG Data], PyG Data, y)

      data_graph_batch: List[List[PyG Data]]
      query_batch     : List[PyG Data]
      Y_batch         : LongTensor [B]
    """
    data_graph_batch, query_batch, y_list = [], [], []
    for sample in batch:

        if isinstance(sample, (tuple, list)) and len(sample) == 2 and isinstance(sample[0], dict):
            x, y = sample
            assert 'data_graph' in x and 'query_graph' in x, \
                f"Expect keys data_graph/query_graph, got {list(x.keys())}"
            data_graph_batch.append(x['data_graph'])
            query_batch.append(x['query_graph'])
            y_list.append(torch.as_tensor(y, dtype=torch.long))

        elif isinstance(sample, (tuple, list)) and len(sample) == 3:
            data_graph_batch.append(sample[0])
            query_batch.append(sample[1])
            y_list.append(torch.as_tensor(sample[2], dtype=torch.long))
        else:
            raise TypeError(f"Unexpected dataset item format: {type(sample)}")

    Y_batch = torch.stack(y_list, dim=0).view(-1)
    return data_graph_batch, query_batch, Y_batch

def create_dataloader(
    dataset,
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: Optional[int] = None,
    sampler=None,
    indices: Optional[Sequence[int]] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
):

    ctx = mp.get_context('spawn') if num_workers and num_workers > 0 else None


    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)


    if indices is not None:
        dataset = Subset(dataset, indices)


    if sampler is not None:
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=custom_collate_fn,      
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        multiprocessing_context=ctx,       
        persistent_workers=(num_workers > 0),
    )
