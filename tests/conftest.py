import pytest

from humanmock.engine import DrugBook, Simulator


@pytest.fixture(scope="session")
def sim():
    return Simulator()


@pytest.fixture(scope="session")
def book():
    return DrugBook.build()
