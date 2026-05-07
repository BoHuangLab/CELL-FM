# -*- coding: utf-8 -*-
from dataclasses import dataclass
from cell_fm.models.cell_fm.cell_fm_config import CELLFMConfig as BaseCELLFMConfig


@dataclass
class CELLFMConfig(BaseCELLFMConfig):
    img_type: str = "GFP"  # "GFP", "SNAP"
