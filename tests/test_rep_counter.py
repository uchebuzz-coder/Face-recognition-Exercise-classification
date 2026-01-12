"""
Unit tests for rep counter
"""

import unittest
from unittest.mock import Mock
from modules.rep_counter import SquatRepCounter, RepState


class TestSquatRepCounter(unittest.TestCase):
    """Test squat rep counter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.counter = SquatRepCounter(down_threshold=90, up_threshold=140)
    
    def test_initial_state(self):
        """Test initial counter state."""
        self.assertEqual(self.counter.get_rep_count(), 0)
        self.assertEqual(self.counter.current_state, RepState.STANDING)
        self.assertEqual(len(self.counter.get_rep_details()), 0)
    
    def test_reset(self):
        """Test counter reset."""
        self.counter.rep_count = 5
        self.counter.reset()
        self.assertEqual(self.counter.get_rep_count(), 0)
        self.assertEqual(self.counter.current_state, RepState.STANDING)


if __name__ == '__main__':
    unittest.main()
