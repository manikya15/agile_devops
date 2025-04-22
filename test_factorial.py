"""Unit tests for the factorial function."""

import unittest
from factorial import factorial

class TestFactorial(unittest.TestCase):
    """Test cases for the factorial function."""

    def test_zero(self):
        """Test that factorial of 0 returns 1."""
        self.assertEqual(factorial(0), 1)

    def test_positive(self):
        """Test that factorial of 5 returns 120."""
        self.assertEqual(factorial(5), 120)

    def test_one(self):
        """Test that factorial of 1 returns 1."""
        self.assertEqual(factorial(1), 1)

    def test_negative(self):
        """Test that factorial raises ValueError for negative input."""
        with self.assertRaises(ValueError):
            factorial(-2)

if __name__ == "__main__":
    unittest.main()
