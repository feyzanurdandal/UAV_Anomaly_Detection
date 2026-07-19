
import os
from pathlib import Path

# src/config.py -> src/ -> proje kökü
PROJECT_ROOT = Path(os.environ.get("UAV_PROJECT_ROOT", Path(__file__).resolve().parent.parent))

MODELS_DIR = Path(os.environ.get("UAV_MODELS_DIR", PROJECT_ROOT / "models"))
DATA_DIR = Path(os.environ.get("UAV_DATA_DIR", PROJECT_ROOT / "data" / "processed"))

ATTACK_POOL_DIR = DATA_DIR / "attack_master_pool"
ALIGNED_FLIGHTS_DIR = DATA_DIR / "aligned_flights"
NORMAL_TEST_DIR = DATA_DIR / "test_normal"
RAW_ULOG_DIR = Path(os.environ.get("UAV_RAW_ULOG_DIR", PROJECT_ROOT / "data" / "raw" / "uav_sead"))

# Model dosyaları
SCALER_PATH = MODELS_DIR / "global_scaler.pkl"
MODEL_WEIGHTS_PATH = MODELS_DIR / "hybrid_model.pth"
THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"


def check_paths(*paths: Path) -> None:
    """Verilen yolların var olup olmadığını kontrol eder; eksikse okunabilir bir hata verir."""
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Aşağıdaki dosya/klasörler bulunamadı:\n  - "
            + "\n  - ".join(missing)
            + f"\n\nBeklenen proje kökü: {PROJECT_ROOT}\n"
            "Farklı bir konumdaysa UAV_PROJECT_ROOT ortam değişkenini ayarlayın, "
            "örn: export UAV_PROJECT_ROOT=/path/to/uav_project"
        )
