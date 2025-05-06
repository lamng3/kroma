import unittest
from algorithms.bisimulation import bisimulation

class TestBisimulation(unittest.TestCase):
    def test_toy_graph(self):
        # toy graph: A→B and C→B, so A and C share identical structure
        graph = {
            'A': {'B'},
            'C': {'B'},
            'B': set()
        }
        pairs = bisimulation(graph)
        # should find A and C in the same block => both (A,C) and (C,A)
        self.assertIn(('A','C'), pairs)
        self.assertIn(('C','A'), pairs)
        # B stands alone (no siblings), so no (B,*) pairs
        for (x,y) in pairs:
            self.assertFalse(x == 'B' or y == 'B')

if __name__ == "__main__":
    unittest.main()
