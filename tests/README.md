# Tests Directory

Unit tests for the workout video analysis system.

## Running Tests

Run all tests:
```bash
python -m unittest discover tests
```

Run specific test file:
```bash
python -m unittest tests.test_utils
python -m unittest tests.test_rep_counter
```

Run specific test:
```bash
python -m unittest tests.test_utils.TestUtils.test_calculate_angle_90_degrees
```

## Test Coverage

- `test_utils.py` - Tests for angle and distance calculations
- `test_rep_counter.py` - Tests for rep counting logic

## Adding Tests

When adding new features, create corresponding test files following the pattern:
```python
import unittest
from modules.your_module import YourClass

class TestYourClass(unittest.TestCase):
    def test_your_feature(self):
        # Test implementation
        pass
```
