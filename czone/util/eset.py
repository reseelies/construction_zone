from collections.abc import Iterable
from collections import deque
import numpy as np
from typing import Union
from functools import reduce

class EqualSet():
    """Creates a set based on equality operations.    
    """
    def __init__(self, x=[]):
        self._storage = deque([])
        self.update(x)

    def __iter__(self):
        yield from iter(self._storage)

    def __len__(self):
        return len(self._storage)
    
    def __contains__(self, item):
        return item in self._storage
    
    @staticmethod
    def _make_unique(x: Iterable):
        match x:
            case EqualSet():
                yield from x
            case _:
                N_items = len(x)
                add_dict = {}
                for i in range(N_items):
                    add_dict[i] = True if i not in add_dict else False
                    if add_dict[i]:
                        for j in range(i+1, N_items):
                            if x[j] == x[i]:
                                add_dict[j] = False
                yield from (val for k, val in zip(sorted(add_dict.keys()), x) if add_dict[k])

    def _check_if_new(self, other):
        for o in other:
            add = True
            for s in self:
                if o == s:
                    add = False
                    break
            if add:
                yield o
    
    def update(self, other: Iterable):
        tmp = self._make_unique(other)
        if len(self) == 0:
            self._storage.extend(tmp)
        else:
            self._storage.extend(self._check_if_new(tmp))

    def union(self, other: Iterable):
        res = EqualSet(self)
        res.update(other)
        return res
    
    def __or__(self, other):
        return self.union(other)
    
    def __ior__(self, other):
        self.update(other)
        return self
    
    def remove(self, other: Iterable):
        tmp = self._make_unique(other)
        for t in tmp:
            try:
                self._storage.remove(t)
            except ValueError:
                pass
    
    def difference(self, other: Iterable):
        res = EqualSet(self)
        res.remove(other)
        return res
    
    def __sub__(self, other):
        return self.difference(other)
    
    def __isub__(self, other):
        self.remove(other)
        return self
    
    def symmetric_difference(self, other: Iterable):
        A = EqualSet(other)
        return (self - A) | (A - self)
    
    def __xor__(self, other: Iterable):
        return self.symmetric_difference(other)
    
    def intersection(self, other: Iterable):
        A = self - other
        return self - A
    
    def __and__(self, other: Iterable):
        return self.intersection(other)

    def isdisjoint(self, other):
        return len(self.intersection(other)) == 0
    
    def issubset(self, other):
        other = EqualSet(other)
        if len(other) >= len(self):
            return reduce(lambda x, y: x and y, [i in other for i in self])
        else:
            return False
    
    def __le__(self, other):
        return self.issubset(other)
    
    def issuperset(self, other):
        other = EqualSet(other)
        if len(other) <= len(self):
            return reduce(lambda x, y: x and y, [i in self for i in other])
        else:
            return False
    
    def __ge__(self, other):
        return self.issuperset(other)

    def __eq__(self, other):
        other = EqualSet(other)
        if len(other) != len(self):
            return False
        else:
            return (self <= other) and (self >= other)

    def __lt__(self, other):
        return (self <= other) and (self != other)
    
    def __gt__(self, other):
        return (self >= other) and (self != other)


def array_set_equal(x, y, **kwargs):
    if x.shape != y.shape:
        return False
    
    xs = np.sort(x, axis=0)
    ys = np.sort(y, axis=0)

    return np.allclose(xs, ys, **kwargs)
