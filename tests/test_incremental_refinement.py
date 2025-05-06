import unittest
from typing import List
from algorithms.bisimulation import compute_node_ranks
from algorithms.incremental_refinement import incremental_refinement, Node, Graph, Edge

class TestIncrementalRefinement(unittest.TestCase):
    def setUp(self):
        # initial compressed graph G_r: A -> B, C isolated
        self.G_r = {
            ('A',''): {('B','')},
            ('B',''): set(),
            ('C',''): set(),
        }
        # compute initial ranks
        self.rank_attr = compute_node_ranks(self.G_r.copy())

    def test_insert_edge_lower_to_higher(self):
        # delta: C -> A (rank C = 0, rank A = 1)
        delta: List[Edge] = [(('C',''), ('A',''))]

        G_updated, expert_queries = incremental_refinement(
            O_S=self.G_r.copy(),
            O_T={},
            G_r=self.G_r.copy(),
            delta_G=delta,
            rank_attr=self.rank_attr.copy()
        )

        # expert_queries should contain exactly one tuple ('C',''),('A','')
        self.assertEqual(len(expert_queries), 1)
        u, up = expert_queries[0]
        self.assertEqual(u, ('C',''))
        self.assertEqual(up, ('A',''))

        # no collapse/merge should have occurred: G_updated equals original
        self.assertDictEqual(G_updated, self.G_r)

if __name__ == "__main__":
    unittest.main()
