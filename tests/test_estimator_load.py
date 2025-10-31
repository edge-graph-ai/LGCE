import pytest

torch = pytest.importorskip("torch")

from src.models.estimator import GraphCardinalityEstimatorMultiSubgraph


def test_old_transformer_decoder_weights_are_ignored():
    model = GraphCardinalityEstimatorMultiSubgraph()
    state = model.state_dict()
    old_key = "transformer_decoder.layers.0.self_attn.in_proj_weight"
    state[old_key] = torch.zeros(1)

    incompatible = model.load_state_dict(state, strict=False)

    unexpected = getattr(incompatible, "unexpected_keys", []) or []
    missing = getattr(incompatible, "missing_keys", []) or []

    assert all(not key.startswith("transformer_decoder.") for key in unexpected)
    assert all(not key.startswith("transformer_decoder.") for key in missing)
