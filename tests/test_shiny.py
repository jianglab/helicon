import unittest
from unittest.mock import MagicMock
from helicon.lib import shiny

class TestShiny(unittest.TestCase):
    def test_get_client_url_query_params(self):
        # Mock the input object
        mock_input = MagicMock()
        mock_input._map = {
            '.clientdata_url_search': MagicMock(return_value='?param1=value1&param2=value2')
        }

        params = shiny.get_client_url_query_params(mock_input)
        self.assertEqual(params, {'param1': ['value1'], 'param2': ['value2']})

        params_no_list = shiny.get_client_url_query_params(mock_input, keep_list=False)
        self.assertEqual(params_no_list, {'param1': 'value1', 'param2': 'value2'})

    def test_set_client_url_query_params(self):
        query_params = {'param1': 'value1', 'param2': 'value2'}
        script_tag = shiny.set_client_url_query_params(query_params)
        self.assertIn("param1=value1&param2=value2", str(script_tag))

if __name__ == '__main__':
    unittest.main()
