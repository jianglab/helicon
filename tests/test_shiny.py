from unittest.mock import MagicMock
from helicon.lib import shiny


class TestShiny(object):
    def test_get_client_url_query_params(self):
        # Mock the input object
        mock_input = MagicMock()
        mock_input._map = {
            ".clientdata_url_search": MagicMock(
                return_value="?param1=value1&param2=value2"
            )
        }

        params = shiny.get_client_url_query_params(mock_input)
        assert params == {"param1": ["value1"], "param2": ["value2"]}

        params_no_list = shiny.get_client_url_query_params(mock_input, keep_list=False)
        assert params_no_list == {"param1": "value1", "param2": "value2"}

    def test_set_client_url_query_params(self):
        query_params = {"param1": "value1", "param2": "value2"}
        script_tag = shiny.set_client_url_query_params(query_params)
        assert "param1=value1&param2=value2" in str(script_tag)
