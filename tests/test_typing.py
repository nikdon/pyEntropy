import inspect
import unittest
from pathlib import Path

import pyentrp


class TestTyping(unittest.TestCase):
    def test_package_is_marked_as_typed(self):
        """Ship the PEP 561 marker so type checkers use the annotations."""
        self.assertTrue((Path(pyentrp.__file__).parent / "py.typed").is_file())

    def test_public_functions_are_fully_annotated(self):
        """Annotate every parameter and the return value of every public function."""
        for name in pyentrp.__all__:
            func = getattr(pyentrp, name)
            signature = inspect.signature(func)
            with self.subTest(name=name):
                self.assertIsNot(signature.return_annotation, inspect.Signature.empty)
                for parameter in signature.parameters.values():
                    self.assertIsNot(parameter.annotation, inspect.Parameter.empty, parameter.name)


if __name__ == "__main__":
    unittest.main()
