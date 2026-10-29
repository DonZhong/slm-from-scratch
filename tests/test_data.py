
import pytest
from slm.data import NextTokenDataset
from slm.data import split_text


def test_split_text_preserves_order() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"

    train_text, val_text = split_text(
        text,
        train_fraction=0.8,
        context_length=4,
    )

    assert train_text == "abcdefghijklmnopqrst"
    assert val_text == "uvwxyz"
    assert train_text + val_text == text


def test_split_text_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError):
        split_text(
            "abcdefghijklmnopqrstuvwxyz",
            train_fraction=1.0,
            context_length=4,
        )


def test_split_text_rejects_short_validation() -> None:
    with pytest.raises(ValueError):
        split_text(
            "abcdefghij",
            train_fraction=0.8,
            context_length=4,
        )

def test_next_token_dataset_default_stride_preserves_behavior():
    dataset = NextTokenDataset(
        token_ids=list(range(20)),
        context_length=4,
    )

    assert len(dataset) == 16

    inputs, targets = dataset[1]

    assert inputs.tolist() == [1, 2, 3, 4]
    assert targets.tolist() == [2, 3, 4, 5]


def test_next_token_dataset_supports_non_overlapping_windows():
    dataset = NextTokenDataset(
        token_ids=list(range(20)),
        context_length=4,
        stride=4,
    )

    assert len(dataset) == 4

    inputs, targets = dataset[0]

    assert inputs.tolist() == [0, 1, 2, 3]
    assert targets.tolist() == [1, 2, 3, 4]

    inputs, targets = dataset[1]

    assert inputs.tolist() == [4, 5, 6, 7]
    assert targets.tolist() == [5, 6, 7, 8]


def test_next_token_dataset_rejects_invalid_stride():
    import pytest

    with pytest.raises(ValueError, match="stride"):
        NextTokenDataset(
            token_ids=list(range(20)),
            context_length=4,
            stride=0,
        )