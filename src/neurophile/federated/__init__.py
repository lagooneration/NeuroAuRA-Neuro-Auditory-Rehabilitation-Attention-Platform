"""
neurophile.federated
====================
Federated machine learning: edge training + server aggregation.

Status: SCAFFOLD — Phase 4

Architecture
------------
- Edge (clinic): trains per-patient AAD model locally; sends weight Δ to server.
- Server: aggregates deltas via FedAvg/FedProx with differential privacy.
- Raw EEG never leaves the edge node.

Modules
-------
edge.py        : Local training loop (Flower client)
server.py      : Aggregation server (Flower server + FedAvg strategy)
privacy.py     : Differential privacy noise injection
protocol.py    : gRPC / REST protocol glue
consent.py     : Per-session opt-in consent management

Dependencies
------------
    pip install "neurophile[federated]"   # installs flwr >= 1.7

References
----------
- McMahan et al. (2017) "Communication-Efficient Learning of Deep Networks
  from Decentralized Data" — FedAvg algorithm.
- Bonawitz et al. (2019) — Secure aggregation.
- Dwork & Roth (2014) — Differential privacy foundations.
"""
