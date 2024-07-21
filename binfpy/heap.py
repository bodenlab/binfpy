import numpy as np

class LabelHeap:
    """
    Min and max heap: data structure for keeping a list of labels, sorted by a value associated with each.
    Based on max heap in Necaise, "Data structures and algorithms in Python" (Ch 13); fixed a bunch of bugs though...
    """
    def __init__(self, maxsize, reverse = False):
        """
        Initialise a heap.
        :param maxsize: the maximum size of the heap
        :param reverse: heap in descending order if true, else ascending
        """
        self.reverse = reverse
        self._elements = np.array([None for _ in range(maxsize)])
        self._idx2val = dict()
        self._count = 0
    def __len__(self):
        """
        The number of elements in the heap currently.
        :return: the number of added elements
        """
        return self._count
    def __str__(self):
        """
        String representation of heap. A list of labels in a binary tree (first element is the smallest/greatest value)
        :return: heap as a string
        """
        return str([y for y in self._elements[:self._count]])
    def __repr__(self):
        return self.__str__()
    def capacity(self):
        """
        Maximum size allocated to heap
        :return: the number of elements that this heap can store
        """
        return len(self._elements)
    def __getitem__(self, i):
        """
        Retrieve the value by tree index (index 0 is the root and contains the smallest/greatest value)
        :param i: index in tree
        :return: the value at this index
        """
        return self._idx2val[self._elements[i]]
    def add(self, label, value):
        """
        Add a label with value to heap
        :param label:
        :param value:
        """
        assert self._count < self.capacity(), "Cannot add to a full heap"
        assert not label in self._idx2val, "Cannot add a duplicate label"
        self._elements[self._count] = label
        self._idx2val[label] = value
        self._count += 1
        self._siftUp(self._count - 1)
    def pop(self):
        """
        Pop the (label, value) pair with minimum/maximum value; removes the entry
        :return: tuple with label and value
        """
        assert self._count > 0, "Cannot extract from an empty heap"
        label = self._elements[0]
        self._count -= 1
        self._elements[0] = self._elements[self._count]
        self._siftDown(0)
        return (label, self._idx2val[label])
    def peek(self):
        """
        Peek the (label, value) pair with minimum/maximum value; does not change the heap
        :return: tuple with label and value
        """
        assert self._count > 0, "Cannot peek in an empty heap"
        return (self._elements[0], self._idx2val[self._elements[0]])
    def _delete(self, i):
        """
        Delete by internal, binary tree index
        :param i: index
        :return:
        """
        assert self._count > i, "Cannot delete index" + str(i)
        self._count -= 1
        self._elements[i] = self._elements[self._count]
        self._siftDown(i)
    def _siftUp(self, i):
        if i > 0:
            parent = (i-1) // 2
            if (self[i] > self[parent] if self.reverse else self[i] < self[parent]): # swap
                tmp = self._elements[i]
                self._elements[i] = self._elements[parent]
                self._elements[parent] = tmp
                self._siftUp(parent)
    def _siftDown(self, i):
        left = 2 * i + 1
        right = 2 * i + 2
        extremist = i
        if left < self._count and (self[left] >= self[extremist] if self.reverse else self[left] <= self[extremist]):
            extremist = left
        if right < self._count and (self[right] >= self[extremist] if self.reverse else self[right] <= self[extremist]):
            extremist = right
        if extremist != i: # swap
            tmp = self._elements[i]
            self._elements[i] = self._elements[extremist]
            self._elements[extremist] = tmp
            self._siftDown(extremist)