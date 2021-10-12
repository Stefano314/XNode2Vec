import numpy as np
from Xnode2vec import complete_edgelist, stellar_edgelist
import pytest

def test_complete_dimension():
    rows = np.random.randint(1,30)
    columns = np.random.randint(1,30)
    dataset = np.random.rand(rows,columns)
    assert len(complete_edgelist(dataset).index) == dataset.shape[0]**2

def test_stellar_dimension():
    rows = np.random.randint(1,30)
    columns = np.random.randint(1,30)
    dataset = np.random.rand(rows,columns)
    assert len(stellar_edgelist(dataset).index) == dataset.shape[0]

def test_complete_zeroweight():
    rows = np.random.randint(1, 30)
    columns = np.random.randint(1, 30)
    dataset = np.zeros((rows, columns))
    assert complete_edgelist(dataset).loc[:,'weight'].values.all() == 1.

def test_stellar_zeroweight():
    rows = np.random.randint(1, 30)
    columns = np.random.randint(1, 30)
    dataset = np.zeros((rows, columns))
    assert complete_edgelist(dataset).loc[:,'weight'].values.all() == 1.


