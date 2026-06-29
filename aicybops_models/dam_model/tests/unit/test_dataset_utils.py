import os
import sys
from pathlib import Path
from unittest import mock

_dam_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_dam_root))
sys.path.insert(0, str(_dam_root / "processing"))

from dataset.dataset_utils import get_dataloader_kwargs


def test_get_dataloader_kwargs_defaults_to_main_process():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("AICYBOPS_DATALOADER_NUM_WORKERS", None)
        os.environ.pop("AICYBOPS_ALLOW_DATALOADER_WORKERS", None)
        assert get_dataloader_kwargs() == {}


def test_get_dataloader_kwargs_disables_workers_in_docker():
    env = {
        "AICYBOPS_DATALOADER_NUM_WORKERS": "2",
        "AICYBOPS_ALLOW_DATALOADER_WORKERS": "",
    }
    with mock.patch.dict(os.environ, env, clear=False):
        with mock.patch("os.path.exists", return_value=True):
            assert get_dataloader_kwargs() == {}


def test_get_dataloader_kwargs_allows_override_in_docker():
    env = {
        "AICYBOPS_DATALOADER_NUM_WORKERS": "2",
        "AICYBOPS_ALLOW_DATALOADER_WORKERS": "1",
    }
    with mock.patch.dict(os.environ, env, clear=False):
        with mock.patch("os.path.exists", return_value=True):
            kwargs = get_dataloader_kwargs()
            assert kwargs["num_workers"] == 2
            assert kwargs["prefetch_factor"] == 2
            assert kwargs["persistent_workers"] is True
