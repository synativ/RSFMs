from rsfms.datasets.fire_scars import FireScarsHLS, FireScarsNonGeo, FireScarsSegmentationMask
from rsfms.datasets.generic_pixel_wise_dataset import (
    GenericNonGeoPixelwiseRegressionDataset,
    GenericNonGeoSegmentationDataset,
)
from rsfms.datasets.generic_scalar_label_dataset import (
    GenericNonGeoClassificationDataset,
)

# GenericNonGeoRegressionDataset
from rsfms.datasets.utils import HLSBands
from rsfms.datasets.sen1floods11 import Sen1Floods11NonGeo

# TorchGeo RasterDatasets
from rsfms.datasets.hls import HLSL30, HLSS30
from rsfms.datasets.wsf import WSF2019, WSFEvolution

# geobench datasets classification
from rsfms.datasets.m_bigearthnet import MBigEarthNonGeo
from rsfms.datasets.m_brick_kiln import MBrickKilnNonGeo
from rsfms.datasets.m_eurosat import MEuroSATNonGeo
from rsfms.datasets.m_forestnet import MForestNetNonGeo
from rsfms.datasets.m_pv4ger import MPv4gerNonGeo
from rsfms.datasets.m_so2sat import MSo2SatNonGeo

# geobench datasets segmentation
from rsfms.datasets.m_cashew_plantation import MBeninSmallHolderCashewsNonGeo
from rsfms.datasets.m_chesapeake_landcover import MChesapeakeLandcoverNonGeo
from rsfms.datasets.m_neontree import MNeonTreeNonGeo
from rsfms.datasets.m_nz_cattle import MNzCattleNonGeo
from rsfms.datasets.m_pv4ger_seg import MPv4gerSegNonGeo
from rsfms.datasets.m_SA_crop_type import MSACropTypeNonGeo


__all__ = (
    "FireScarsNonGeo",
    "FireScarsHLS",
    "FireScarsSegmentationMask",
    "GenericNonGeoSegmentationDataset",
    "GenericNonGeoPixelwiseRegressionDataset",
    "GenericNonGeoClassificationDataset",
    "GenericNonGeoRegressionDataset",
    "HLSBands",
    "Sen1Floods11NonGeo",
    "HLSL30",
    "HLSS30",
    "WSF2019",
    "WSFEvolution",
    "MBigEarthNonGeo",
    "MBrickKilnNonGeo",
    "MEuroSATNonGeo",
    "MForestNetNonGeo",
    "MPv4gerNonGeo",
    "MSo2SatNonGeo",
    "MBeninSmallHolderCashewsNonGeo",
    "MChesapeakeLandcoverNonGeo",
    "MNeonTreeNonGeo",
    "MNzCattleNonGeo",
    "MPv4gerSegNonGeo",
    "MSACropTypeNonGeo",
)
