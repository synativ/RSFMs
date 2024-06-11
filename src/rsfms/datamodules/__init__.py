from rsfms.datamodules.fire_scars import FireScarsNonGeoDataModule
from rsfms.datamodules.generic_pixel_wise_data_module import (
    GenericNonGeoPixelwiseRegressionDataModule,
    GenericNonGeoSegmentationDataModule,
)
from rsfms.datamodules.generic_scalar_label_data_module import (
    GenericNonGeoClassificationDataModule,
)

# GenericNonGeoRegressionDataModule,
from rsfms.datamodules.sen1floods11 import Sen1Floods11NonGeoDataModule
from rsfms.datamodules.torchgeo_data_module import TorchGeoDataModule, TorchNonGeoDataModule

# geobench classification datamodules
from rsfms.datamodules.m_bigearthnet import MBigEarthNonGeoDataModule
from rsfms.datamodules.m_brick_kiln import MBrickKilnNonGeoDataModule
from rsfms.datamodules.m_eurosat import MEuroSATNonGeoDataModule
from rsfms.datamodules.m_forestnet import MForestNetNonGeoDataModule
from rsfms.datamodules.m_pv4ger import MPv4gerNonGeoDataModule
from rsfms.datamodules.m_so2sat import MSo2SatNonGeoDataModule

# geobench segmentation datamodules
from rsfms.datamodules.m_cashew_plantation import MBeninSmallHolderCashewsNonGeoDataModule
from rsfms.datamodules.m_chesapeake_landcover import MChesapeakeLandcoverNonGeoDataModule
from rsfms.datamodules.m_neontree import MNeonTreeNonGeoDataModule
from rsfms.datamodules.m_nz_cattle import MNzCattleNonGeoDataModule
from rsfms.datamodules.m_pv4ger_seg import MPv4gerSegNonGeoDataModule
from rsfms.datamodules.m_SA_crop_type import MSACropTypeNonGeoDataModule


__all__ = (
    "FireScarsNonGeoDataModule",
    "GenericNonGeoSegmentationDataModule",
    "GenericNonGeoPixelwiseRegressionDataModule",
    "GenericNonGeoSegmentationDataModule",
    "GenericNonGeoClassificationDataModule",
    # "GenericNonGeoRegressionDataModule",
    "Sen1Floods11NonGeoDataModule",
    "TorchGeoDataModule",
    "TorchNonGeoDataModule",
    "MBigEarthNonGeoDataModule",
    "MBrickKilnNonGeoDataModule",
    "MEuroSATNonGeoDataModule",
    "MForestNetNonGeoDataModule",
    "MPv4gerNonGeoDataModule",
    "MSo2SatNonGeoDataModule",
    "MBeninSmallHolderCashewsNonGeoDataModule",
    "MChesapeakeLandcoverNonGeoDataModule",
    "MNeonTreeNonGeoDataModule",
    "MNzCattleNonGeoDataModule",
    "MPv4gerSegNonGeoDataModule",
    "MSACropTypeNonGeoDataModule",
)
