"""
Unit tests for utility functions
"""

import unittest
import numpy as np
from modules.utils import calculate_angle, calculate_distance


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_calculate_angle_90_degrees(self):
        """Test 90 degree angle calculation."""
        point1 = (1.0, 0.0, 0.0)
        point2 = (0.0, 0.0, 0.0)
        point3 = (0.0, 1.0, 0.0)
        
        angle = calculate_angle(point1, point2, point3)
        self.assertAlmostEqual(angle, 90.0, places=1)
    
    def test_calculate_angle_180_degrees(self):
        """Test 180 degree angle calculation."""
        point1 = (1.0, 0.0, 0.0)
        point2 = (0.0, 0.0, 0.0)
        point3 = (-1.0, 0.0, 0.0)
        
        angle = calculate_angle(point1, point2, point3)
        self.assertAlmostEqual(angle, 180.0, places=1)
    
    def test_calculate_angle_0_degrees(self):
        """Test 0 degree angle calculation."""
        point1 = (1.0, 0.0, 0.0)
        point2 = (0.0, 0.0, 0.0)
        point3 = (2.0, 0.0, 0.0)
        
        angle = calculate_angle(point1, point2, point3)
        self.assertAlmostEqual(angle, 0.0, places=1)
    
    def test_calculate_distance(self):
        """Test distance calculation."""
        point1 = (0.0, 0.0, 0.0)
        point2 = (3.0, 4.0, 0.0)
        
        distance = calculate_distance(point1, point2)
        self.assertAlmostEqual(distance, 5.0, places=1)
    
    def test_calculate_distance_3d(self):
        """Test 3D distance calculation."""
        point1 = (0.0, 0.0, 0.0)
        point2 = (1.0, 1.0, 1.0)
        
        distance = calculate_distance(point1, point2)
        self.assertAlmostEqual(distance, np.sqrt(3), places=4)


if __name__ == '__main__':
    unittest.main()
