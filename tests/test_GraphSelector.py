import random

from src.GraphSelector import GraphSelector

SECRET_KEY1 = "1234"
SECRET_KEY2 = "6789"

def test_stores_secret_key():
    selector = GraphSelector(secret_key=SECRET_KEY1)
    assert selector.secret_key == SECRET_KEY1
 
def test_default_percentage():
    selector = GraphSelector(secret_key=SECRET_KEY1)
    assert selector.percentage == 0.05
 
def test_custom_percentage():
    selector = GraphSelector(secret_key=SECRET_KEY1, percentage=0.1)
    assert selector.percentage == 0.1

def test_same_key_same_shuffle():
    rng1 = random.Random(SECRET_KEY1)
    rng2 = random.Random(SECRET_KEY1)

    indices1 = list(range(100))
    indices2 = list(range(100))

    rng1.shuffle(indices1)
    rng2.shuffle(indices2)

    assert indices1 == indices2

def test_different_keys_different_shuffles():
    rng1 = random.Random(SECRET_KEY1)
    rng2 = random.Random(SECRET_KEY2)

    indices1 = list(range(100))
    indices2 = list(range(100))

    rng1.shuffle(indices1)
    rng2.shuffle(indices2)

    assert indices1 != indices2