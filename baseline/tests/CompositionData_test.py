import pytest

from lib.config import AppConfig, DataSource
from lib.data_handling import CompositionData
from lib.reproduction import major_oxides


@pytest.mark.parametrize("data_source", [DataSource.PDS, DataSource.CCAM])
def test_composition_data(data_source):
    config = AppConfig(data_source=data_source)
    composition_data_df = CompositionData(
        config.data_source, config.composition_data_path
    ).composition_data

    # Check that the dataframe has at least one row and that all major oxides are columns
    assert not composition_data_df.empty and all(
        oxide in composition_data_df.columns for oxide in major_oxides
    )
