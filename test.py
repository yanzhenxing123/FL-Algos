"""

test
"""
from Flower_demo import client, server
import unittest

import unittest


class IntegerArithmeticTestCase(unittest.TestCase):
    def test_server(self):
        server.main()

    def test_client(self):  # test method names begin with 'test'
        client.main()

    def testMultiply(self):
        self.assertEqual((0 * 10), 0)
        self.assertEqual((5 * 8), 40)


if __name__ == '__main__':
    unittest.main()
