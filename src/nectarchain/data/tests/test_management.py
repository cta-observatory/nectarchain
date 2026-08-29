import os
from pathlib import Path
from unittest.mock import patch

import pytest

from nectarchain.data.management import DataManagement


class TestDataManagement:
    @patch("nectarchain.data.management.glob.glob")
    @patch.dict(os.environ, {"NECTARCAMDATA": "/test/data"}, clear=True)
    def test_findrun_finds_files(self, mock_glob):
        mock_glob.return_value = [
            "/test/data/runs/NectarCAM.Run0042.0.fits.fz",
            "/test/data/runs/NectarCAM.Run0042.1.fits.fz",
        ]
        name, list_path = DataManagement.findrun(42, search_on_GRID=False)
        assert isinstance(name, Path)
        assert len(list_path) == 2

    @patch("nectarchain.data.management.glob.glob")
    @patch.dict(os.environ, {"NECTARCAMDATA": "/test/data"}, clear=True)
    def test_findrun_no_files_raises(self, mock_glob):
        mock_glob.return_value = []
        with pytest.raises(FileNotFoundError, match="run 42 is not present"):
            DataManagement.findrun(42, search_on_GRID=False)

    @patch("nectarchain.data.management.glob.glob")
    @patch.dict(os.environ, {"NECTARCAMDATA": "/test/data"}, clear=True)
    @patch.object(DataManagement, "get_GRID_location")
    @patch.object(DataManagement, "getRunFromDIRAC")
    def test_findrun_search_on_grid(self, mock_get_run, mock_get_grid, mock_glob):
        mock_glob.side_effect = [[], ["/test/data/runs/NectarCAM.Run0042.0.fits.fz"]]
        mock_get_grid.return_value = ["/lfn/run.fits.fz"]
        name, list_path = DataManagement.findrun(42, search_on_GRID=True)
        assert mock_get_grid.call_count == 2
        mock_get_run.assert_called_once()

    @patch("nectarchain.data.management.glob.glob")
    @patch.dict(os.environ, {"NECTARCAMDATA": "/test/data"}, clear=True)
    @patch.object(DataManagement, "get_GRID_location")
    def test_findrun_grid_no_lfns_raises(self, mock_get_grid, mock_glob):
        mock_glob.side_effect = [[], []]
        mock_get_grid.return_value = []
        with pytest.raises(FileNotFoundError, match="Could not find run 42 on DIRAC"):
            DataManagement.findrun(42, search_on_GRID=True)

    @patch("nectarchain.data.management.glob.glob")
    @patch.dict(os.environ, {"NECTARCAMDATA": "/test/data"}, clear=True)
    def test_findrun_returns_sorted(self, mock_glob):
        mock_glob.return_value = [
            "/test/data/runs/NectarCAM.Run0042.2.fits.fz",
            "/test/data/runs/NectarCAM.Run0042.1.fits.fz",
        ]
        name, list_path = DataManagement.findrun(42, search_on_GRID=False)
        suffixes = [p.suffixes[1] for p in list_path]
        assert suffixes == sorted(suffixes)

    @patch("nectarchain.data.management.glob.glob")
    @patch.dict(os.environ, {"NECTARCAMDATA": "/test/data"}, clear=True)
    def test_find_waveforms(self, mock_glob):
        mock_glob.return_value = ["/test/data/runs/waveforms/test_run42.h5"]
        assert DataManagement.find_waveforms(42) == [
            "/test/data/runs/waveforms/test_run42.h5"
        ]

    @patch("nectarchain.data.management.glob.glob")
    @patch.dict(os.environ, {"NECTARCAMDATA": "/test/data"}, clear=True)
    def test_find_waveforms_no_files(self, mock_glob):
        mock_glob.return_value = []
        with pytest.raises(FileNotFoundError):
            DataManagement.find_waveforms(42)

    @patch("nectarchain.data.management.glob.glob")
    @patch.dict(os.environ, {"NECTARCAMDATA": "/test/data"}, clear=True)
    def test_find_waveforms_multiple_raises(self, mock_glob):
        mock_glob.return_value = ["a.h5", "b.h5"]
        with pytest.raises(FileExistsError):
            DataManagement.find_waveforms(42)

    @patch("nectarchain.data.management.glob.glob")
    @patch.dict(os.environ, {"NECTARCAMDATA": "/test/data"}, clear=True)
    def test_find_charges(self, mock_glob):
        p = "/test/data/runs/charges/test_run42_FullWaveformSum_default.h5"
        mock_glob.return_value = [p]
        assert DataManagement.find_charges(42) == [p]

    @patch("nectarchain.data.management.glob.glob")
    @patch.dict(os.environ, {"NECTARCAMDATA": "/test/data"}, clear=True)
    def test_find_photostat(self, mock_glob):
        p = (
            "/test/data/PhotoStat/PhotoStatisticNectarCAM_FFrun1_"
            "FullWaveformSum_default_Pedrun2.h5"
        )
        mock_glob.return_value = [p]
        assert DataManagement.find_photostat(1, 2) == [p]

    @patch("nectarchain.data.management.glob.glob")
    @patch.dict(os.environ, {"NECTARCAMDATA": "/test/data"}, clear=True)
    def test_find_photostat_no_file(self, mock_glob):
        mock_glob.return_value = []
        with pytest.raises(FileNotFoundError):
            DataManagement.find_photostat(1, 2)

    @patch("nectarchain.data.management.glob.glob")
    @patch.dict(os.environ, {"NECTARCAMDATA": "/test/data"}, clear=True)
    def test_find_SPE_HHV(self, mock_glob):
        p = (
            "/test/data/SPEfit/FlatFieldSPEHHVNectarCAM_run42_"
            "FullWaveformSum_default.h5"
        )
        mock_glob.return_value = [p]
        assert DataManagement.find_SPE_HHV(42) == [p]

    @patch("nectarchain.data.management.glob.glob")
    @patch.dict(os.environ, {"NECTARCAMDATA": "/test/data"}, clear=True)
    def test_find_SPE_HHV_no_file(self, mock_glob):
        mock_glob.return_value = []
        with pytest.raises(FileNotFoundError):
            DataManagement.find_SPE_HHV(42)

    @patch("nectarchain.data.management.glob.glob")
    @patch.dict(os.environ, {"NECTARCAMDATA": "/test/data"}, clear=True)
    def test_find_SPE_HHV_selects_non_maxevents(self, mock_glob):
        mock_glob.return_value = [
            "/test/data/SPEfit/run42_maxevents50.h5",
            "/test/data/SPEfit/run42.h5",
        ]
        result = DataManagement.find_SPE_HHV(42)
        assert "maxevents" not in result[0]

    @patch("nectarchain.data.management.glob.glob")
    @patch.dict(os.environ, {"NECTARCAMDATA": "/test/data"}, clear=True)
    def test_find_SPE_nominal(self, mock_glob):
        p = "/test/data/SPEfit/FlatFieldSPENominalStdNectarCAM_run42.h5"
        mock_glob.return_value = [p]
        assert len(DataManagement.find_SPE_nominal(42)) > 0

    @patch("nectarchain.data.management.glob.glob")
    @patch.dict(os.environ, {"NECTARCAMDATA": "/test/data"}, clear=True)
    def test_find_SPE_combined(self, mock_glob):
        p = "/test/data/SPEfit/FlatFieldCombinedNectarCAM_run42.h5"
        mock_glob.return_value = [p]
        assert len(DataManagement.find_SPE_combined(42)) > 0
