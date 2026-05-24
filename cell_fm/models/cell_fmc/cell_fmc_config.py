from dataclasses import dataclass
from cell_fm.models.cell_fm.cell_fm_config import CELLFMConfig


@dataclass
class CELLFMCConfig(CELLFMConfig):
    model_type: str = 'cell_fmc'

    # Interaction context
    n_context_proteins: int = 474
    context_embedding_dim: int = 1152

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
