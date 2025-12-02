# LGCE: An Efficient and Deployable Learned Graph Cardinality Estimator

This repository contains the reference implementation of **LGCE**, a learned cardinality estimator for pattern-matching Cypher queries over property graphs (Neo4j).

---

## Overview

LGCE models a Cypher pattern query as a **query graph** and the database as a **data graph**. The data graph is partitioned into multiple subgraphs; a lightweight Graph Isomorphism Network (GIN) encodes both the query graph and each data subgraph into a shared embedding space. A self-attention layer captures dependencies across data subgraphs, and a cross-attention layer allows the query representation to attend to and aggregate the most relevant subgraphs in a retrieval-style manner for cardinality prediction.

The implementation in this repository is the one used for all experiments in the paper.

---

## Environment & Installation

Tested with:

- Python ≥ 3.10
- PyTorch ≥ 2.0
- CUDA 12.x
- PyTorch Geometric

Clone the repository and install dependencies:
- git clone https://github.com/edge-graph-ai/LGCE.git
- cd LGCE
- pip install -r requirements.txt

---
## Train

- cd src
- bash run_train.sh

This repository provides a prepared **WordNet** dataset so that you can directly train the model and obtain results on this dataset.  
For more datasets, please refer to: https://github.com/RapidsAtHKUST/SubgraphMatching

---
## Training Details

Optimizer: AdamW  

Initial learning rate: 1e-4  

Batch size: 8 

Max epochs: 50  

Early stopping: patience of 10 epochs  

Random seed: in our reference scripts, we fix the seeds for Python / NumPy / PyTorch to 42.






