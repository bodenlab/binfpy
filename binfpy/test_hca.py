import unittest

from binfpy.hca import *
import random


class MyTestCase(unittest.TestCase):

    N = 8

    def setUp(self):
        """Set up for each test"""
        self.pairidxs1 = dict()
        y = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                self.pairidxs1[(i, j)] = y
                y += 1
        self.pairidxs2 = dict()
        for i in range(self.N):
            for j in range(0, i):
                self.pairidxs2[(i, j)] = self.pairidxs1[(j, i)]

    def test_PairArray1(self):
        pa1 = PairArray(self.N)
        pa2 = PairArray(self.N)
        for p in self.pairidxs1:
            pa1[p] = self.pairidxs1[p]
        for p in self.pairidxs2:
            pa2[p] = self.pairidxs2[p]
        for i, j in self.pairidxs1:
            self.assertEqual(pa1[(j, i)], self.pairidxs1[(i, j)])
        for i, j in self.pairidxs2:
            self.assertEqual(pa2[(j, i)], pa1[(j, i)])

    def test_DNode1(self):
        layer0 = [DNode(i) for i in range(0, 10)]
        layer1 = []
        for i in range(0, len(layer0) // 2):
            layer1.append(
                DNode(
                    i + len(layer0),
                    children=[layer0[i * 2], layer0[i * 2 + 1]],
                    dist=random.randint(1, 10),
                )
            )
        root = DNode(len(layer0) + len(layer1), layer1, dist=100)
        self.assertEquals(root.nChildren(), len(layer1))
        self.assertEquals(len(root.getLeaves()), len(layer0))
        for i in range(len(layer1)):
            self.assertEquals(layer1[i].nChildren(), 2)
        for i in range(len(layer0)):
            self.assertEquals(layer0[i].nChildren(), 0)

    def test_DNode2(self):
        layer0 = [DNode(i) for i in range(0, 10)]
        layer1 = []
        for i in range(0, len(layer0) // 2):
            layer1.append(
                DNode(
                    i + len(layer0),
                    children=[layer0[i * 2], layer0[i * 2 + 1]],
                    dist=random.randint(1, 10),
                )
            )
        root1 = DNode(len(layer0) + len(layer1), layer1, dist=100)
        s1 = str(root1)
        root2 = parse(s1)
        self.assertEquals(root2.nChildren(), root1.nChildren())
        self.assertEquals(len(root2.getLeaves()), len(root1.getLeaves()))
        s2 = str(root2)
        root3 = parse(s2)
        self.assertEquals(str(root3), s2)

    def test_DNode3(self):
        layer0 = [DNode(i) for i in range(0, 8)]
        layer1 = []
        for i in range(0, len(layer0) // 2):
            layer1.append(
                DNode(
                    i + len(layer0),
                    children=[layer0[i * 2], layer0[i * 2 + 1]],
                    dist=random.randint(1, 10),
                )
            )
        layer2 = []
        for i in range(0, len(layer1) // 2):
            layer2.append(
                DNode(
                    i + len(layer0) + len(layer1),
                    children=[layer1[i * 2], layer1[i * 2 + 1]],
                    dist=random.randint(11, 20),
                )
            )
        root = DNode(len(layer0) + len(layer1) + len(layer2), layer2, dist=30)
        chars = "ABCDEFGHIJKLMNOP"
        labels_list = [ch for ch in chars]
        root1 = parse(root.newick(labels_list))
        labels_rev = [ch for ch in chars[::-1]]
        labels_off1 = [ch for ch in chars[1:]]
        labels_dict = {}
        for i in range(len(labels_list)):
            labels_dict[i] = labels_list[i]
        root2 = parse(root.newick(labels_dict))
        self.assertEquals(
            len(parse(root.newick(labels_rev)).getLeaves()), len(root.getLeaves())
        )
        self.assertEquals(root.newick(labels_dict), root.newick(labels_list))
        for ch in chars[:-1]:  # all chars except last one
            node1 = root1.findNode(ch)
            node2 = root2.findNode(ch)
            self.assertIsNotNone(node1)
            self.assertIsNotNone(node2)
            self.assertEquals(len(node1.getLeaves()), len(node2.getLeaves()))
            self.assertEquals(str(root1.findNode(ch)), str(root2.findNode(ch)))

    def test_DNode4(self):
        pass


if __name__ == "__main__":
    unittest.main()
