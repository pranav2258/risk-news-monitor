import pytest
from helpers import DATA

from monitor.data import Store


@pytest.fixture
def store() -> Store:
    s = Store(":memory:")
    s.load_counterparties(DATA / "counterparties.csv")
    return s
