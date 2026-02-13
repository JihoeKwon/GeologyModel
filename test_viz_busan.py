"""Busan 시각화 테스트 - body_geometry 기반 렌더링 검증"""
import sys
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# 1) KB 먼저 로드 (sys.modules에 등록)
from geology_agent.data_registry import check_region_data, get_base_dir

base = get_base_dir()
info = check_region_data("부산", base)
paths = info["paths"]

kb_path = paths.get("knowledge_base_path")
if kb_path:
    spec = importlib.util.spec_from_file_location("busan_geological_knowledge_auto", kb_path)
    kb_mod = importlib.util.module_from_spec(spec)
    sys.modules["busan_geological_knowledge_auto"] = kb_mod
    spec.loader.exec_module(kb_mod)
    print(f"KB loaded: {len(kb_mod.ROCK_UNITS)} rock units")
    for code, info_dict in kb_mod.ROCK_UNITS.items():
        geo = info_dict.get("body_geometry", "MISSING")
        print(f"  {code}: body_geometry={geo}")

# 2) 레거시 모듈을 임포트 (CONFIG가 init_config()로 초기화됨)
import apply_llm_results

# 3) 임포트된 후 reconfigure()로 부산 경로 패치
from config_loader import reconfigure
reconfigure(
    data_dir=paths["geology_dir"],
    geology_dir=paths["geology_dir"],
    dem_file=paths.get("dem_file"),
    dem_dir=paths.get("dem_dir"),
    output_dir=paths["output_dir"],
    target_crs=paths["crs"],
    shapefiles=paths["shapefiles"],
    region_name_kr=paths["region_name_kr"],
    region_name_en=paths["region_name_en"],
)

# 4) body_geometry 캐시 초기화 (이전 실행 잔여 방지)
apply_llm_results._BODY_GEOMETRY_CACHE.clear()

# 5) 실행
apply_llm_results.main()
