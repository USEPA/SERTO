# python imports
import unittest

# third party imports

# local imports
from serto import parse_args, configured_parser


class TestPlumeCLI(unittest.TestCase):

    def setUp(self):
        pass

    def test_plume_cli(self):
        """
        Test the plume cli
        :return:
        """
        parse_args(
            parser=configured_parser(),
            args=[
                'analysis',
                'plumes',
                '--help',
            ]
        )
