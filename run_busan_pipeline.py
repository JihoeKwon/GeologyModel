"""Busan 전체 파이프라인: LLM 분석 → 단면도 생성"""
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

# 2) LLM 분석 모듈 임포트
import geologist_agent_llm

# 3) reconfigure로 부산 경로 패치
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

# 4) 부산 KB를 LLM 모듈에 다시 로드
geologist_agent_llm.load_knowledge_base(str(paths["geology_dir"]))

# 5) LLM 분석 실행
print("\n" + "=" * 70)
print("  STEP 1: LLM Geologist Analysis (부산)")
print("=" * 70)
geologist_agent_llm.main()

# 6) 단면도 생성
print("\n" + "=" * 70)
print("  STEP 2: Cross-Section Visualization (부산)")
print("=" * 70)
import apply_llm_results
# apply_llm_results import 시 init_config()가 기본 경로로 초기화되므로 다시 패치
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
apply_llm_results._BODY_GEOMETRY_CACHE.clear()
apply_llm_results.main()
