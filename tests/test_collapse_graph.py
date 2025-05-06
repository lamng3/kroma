import unittest
from algorithms.bisimulation import collapse

class TestCollapse(unittest.TestCase):
    def test_collapse_simple(self):
        # Graph:
        #   A → B
        #   C → A
        graph = {
            'A': {'B'},
            'B': set(),
            'C': {'A'}
        }
        # Collapse block {'A','B'} into rep='A' (iteration order)
        collapse(graph, {'A','B'})

        # After collapse, 'B' should be gone
        self.assertNotIn('B', graph)

        # 'A' should now have any B‑children rerouted (none), and keep existing children
        self.assertEqual(graph['A'], set())

        # Parents pointing to B (none) would have been rerouted to A.
        # C pointed to A originally; it should remain so.
        self.assertEqual(graph['C'], {'A'})

    def test_collapse_with_external_children(self):
        # More complex:
        #   A → {B, D}
        #   B → {E}
        #   C → {B}
        graph = {
            'A': {'B', 'D'},
            'B': {'E'},
            'C': {'B'},
            'D': set(),
            'E': set(),
        }
        collapse(graph, {'A','B'})

        # B removed
        self.assertNotIn('B', graph)

        # A should now have original D plus B’s child E
        self.assertEqual(graph['A'], {'D', 'E'})

        # Any parent that had B as child (C) should now point to A
        self.assertEqual(graph['C'], {'A'})

    def test_collapse_empty_block(self):
        graph = {'X': {'Y'}, 'Y': set()}
        # collapsing empty block should do nothing
        collapse(graph, set())
        self.assertIn('X', graph)
        self.assertIn('Y', graph)
        self.assertEqual(graph['X'], {'Y'})

if __name__ == '__main__':
    unittest.main()
