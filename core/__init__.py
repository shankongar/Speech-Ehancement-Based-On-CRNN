try:
    from core.utils import *
    from core.model import CRNNSpeechEnhancement
    from core.dataset import SpeechEnhancementDataset, collate_fn
except ImportError as e:
    import warnings
    warnings.warn(f"⚠️ 核心模块导入失败: {e}\n   请确保已安装所有依赖: pip install -r requirements.txt")
    raise
