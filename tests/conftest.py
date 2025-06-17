import pytest

from tests.compare_utils import get_param_path
from tests.config import consts


@pytest.fixture
def param_path():
    get_param_path(consts.model_preset)
