
import pytest

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
