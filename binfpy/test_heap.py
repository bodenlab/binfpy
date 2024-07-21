import unittest
from binfpy.heap import *
import random


class MyTestCase(unittest.TestCase):

    def setUp(self):
        """Set up for each test"""
        idxs = [i for i in range(random.randint(0, 10), random.randint(10, 50))]
        random.shuffle(idxs)
        self.a = [(idx, random.random()) for idx in idxs]
        self.mh = LabelHeap(len(self.a))
        self.maxh = LabelHeap(len(self.a), reverse=True)
        for address, value in self.a:
            self.mh.add(address, value)
            self.maxh.add(address, value)

    def test_MinHeap1(self):
        self.assertEqual(len(self.mh), len(self.a))

    def test_MinHeap2(self):
        minidx = 0
        for i in range(1, len(self.a)):
            if self.a[i][1] < self.a[minidx][1]:
                minidx = i
        # print(self.mh._elements[0], self.mh[0])
        (address, value) = self.mh.pop()
        self.assertEqual(address, self.a[minidx][0])
        self.assertEqual(value, self.a[minidx][1])

    def test_MinHeap3(self):
        ys = [y[1] for y in self.a]
        ys.sort(reverse=False)
        for y in ys:
            self.assertEqual(y, self.mh[0])
            self.mh.pop()

    def test_MaxHeap3(self):
        ys = [y[1] for y in self.a]
        ys.sort(reverse=True)
        for y in ys:
            self.assertEqual(y, self.maxh[0])
            self.maxh.pop()

    def test_MinHeap4(self):
        mh1 = LabelHeap(10)
        self.assertEquals(len(mh1), 0)
        mh1.add("a", 2)
        self.assertEquals(len(mh1), 1)
        mh1.add("b", 1)
        self.assertEquals(len(mh1), 2)
        (label, y) = mh1.pop()
        self.assertEquals(label, "b")
        self.assertEquals(len(mh1), 1)
        mh1.add("c", 3)
        self.assertEquals(len(mh1), 2)


if __name__ == "__main__":
    unittest.main()
