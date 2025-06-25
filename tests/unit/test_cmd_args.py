import yaml
from unittest import mock
from tempfile import NamedTemporaryFile
from src.utils.cmd_args import parse_args


def test_parse_args_valid_config():
    """Test with a valid config file."""
    with NamedTemporaryFile("w") as temp_file:
        yaml.dump({"key": "value"}, temp_file)
        temp_file.flush()

        with mock.patch("argparse._sys.argv", ["script.py", "--cfg", temp_file.name]):
            config = parse_args()

        assert config["key"] == "value"
