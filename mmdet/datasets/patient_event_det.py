import mmcv
import numpy as np
from json import load
from tqdm import trange
from contracts import contract
from os import path as osp

from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class PatientEventDet(CocoDataset):

    CLASSES = (
        "patient_resting", 
        "patient_sitting", 
        "patient_getting_up", 
        "patient_standing", 
        "patient_fell", 
        "staff",
        "visitor"
    )


        



