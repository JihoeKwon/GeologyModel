"""
Data Registry for Geology Agent
지역별 데이터 등록 및 조회
"""

from pathlib import Path
from typing import Dict, Optional, Any
import os


# 지역별 데이터 등록
REGIONS: Dict[str, Dict[str, Any]] = {
    "서울": {
        "name_kr": "서울",
        "name_en": "Seoul",
        "path": "DataRepository/SeoulData",
        "pdf": "지질도_도폭설명서_5만축척_FG33_서울.pdf",
        "geology_dir": "수치지질도_5만축척_FG33_서울",
        "dem_dir": "DEM5m_2018_서울_5186",
        "dem_file": "DEM5m_2018_서울.img",
        "knowledge_base": "seoul_geological_knowledge.py",
        "shapefiles": {
            "litho": "FG33_Geology_50K_Litho.shp",
            "boundary": "FG33_Geology_50K_Boundary.shp",
            "fault": "FG33_Geology_50K_Fault.shp",
            "foliation": "FG33_Geology_50K_Foliation.shp",
            "crosssection": "FG33_Geology_50K_Crosssectionline.shp",
            "frame": "FG33_Geology_50K_Frame.shp",
        },
        "crs": "EPSG:5186",
    },
    "부산": {
        "name_kr": "부산",
        "name_en": "Busan",
        "path": "DataRepository/BusanData",
        "pdf": "지질도_도폭설명서_5만축척_IE00_부산.pdf",
        "geology_dir": "수치지질도_5만축척_IE00_부산",
        "dem_dir": "(B080)공개DEM_35913_img_2025",
        "dem_file": "35913.img",
        "knowledge_base": "busan_geological_knowledge_auto.py",
        "shapefiles": {
            "litho": "IE00_Geology_50K_Litho.shp",
            "boundary": "IE00_Geology_50K_Boundary.shp",
            "fault": "IE00_Geology_50K_Fault.shp",
            "foliation": "IE00_Geology_50K_Foliation.shp",
            "crosssection": "IE00_Geology_50K_Crosssectionline.shp",
            "frame": "IE00_Geology_50K_Frame.shp",
        },
        "crs": "EPSG:5179",
    },
}

# 지역명 별칭 매핑 (여러 표현 지원)
REGION_ALIASES: Dict[str, str] = {
    "seoul": "서울",
    "Seoul": "서울",
    "SEOUL": "서울",
    "busan": "부산",
    "Busan": "부산",
    "BUSAN": "부산",
}


def normalize_region_name(region: str) -> str:
    """지역명 정규화 (별칭 처리)"""
    return REGION_ALIASES.get(region, region)


def get_base_dir() -> Path:
    """프로젝트 기본 디렉토리 반환"""
    # geology_agent 패키지 위치 기준으로 상위 디렉토리
    return Path(__file__).parent.parent


def check_region_data(region: str, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    지역 데이터 존재 여부 확인

    Args:
        region: 지역명 (예: "서울", "Seoul")
        base_dir: 기본 디렉토리 (기본값: 프로젝트 루트)

    Returns:
        {
            "available": bool,
            "region": str,
            "paths": dict (경로 정보),
            "missing": list (누락된 파일/디렉토리),
            "message": str (상태 메시지)
        }
    """
    base_dir = base_dir or get_base_dir()
    region = normalize_region_name(region)

    result = {
        "available": False,
        "region": region,
        "paths": {},
        "missing": [],
        "message": "",
    }

    # 등록된 지역인지 확인
    if region not in REGIONS:
        result["message"] = f"지역 '{region}'이(가) 등록되어 있지 않습니다. 가능한 지역: {list(REGIONS.keys())}"
        return result

    region_config = REGIONS[region]
    data_dir = base_dir / region_config["path"]

    # 데이터 디렉토리 확인
    if not data_dir.exists():
        result["missing"].append(str(data_dir))
        result["message"] = f"데이터 디렉토리가 없습니다: {data_dir}"
        return result

    result["paths"]["data_dir"] = str(data_dir)

    # PDF 파일 확인 (선택적)
    pdf_path = data_dir / region_config["pdf"]
    if pdf_path.exists():
        result["paths"]["pdf_path"] = str(pdf_path)
    else:
        result["paths"]["pdf_path"] = None
        # PDF는 선택적이므로 missing에 추가하지 않음

    # 지질 데이터 디렉토리 확인
    geology_dir = data_dir / region_config["geology_dir"]
    if not geology_dir.exists():
        result["missing"].append(str(geology_dir))
    else:
        result["paths"]["geology_dir"] = str(geology_dir)

        # Shapefile 확인
        for shp_name, shp_file in region_config["shapefiles"].items():
            shp_path = geology_dir / shp_file
            if not shp_path.exists():
                result["missing"].append(f"{shp_name}: {shp_path}")

    # DEM 디렉토리 확인
    dem_dir = data_dir / region_config["dem_dir"]
    if not dem_dir.exists():
        result["missing"].append(str(dem_dir))
    else:
        result["paths"]["dem_dir"] = str(dem_dir)

        # DEM 파일 확인
        dem_file = dem_dir / region_config["dem_file"]
        if not dem_file.exists():
            result["missing"].append(str(dem_file))
        else:
            result["paths"]["dem_file"] = str(dem_file)

    # 지식베이스 확인 (선택적)
    kb_path = data_dir / region_config["knowledge_base"]
    if kb_path.exists():
        result["paths"]["knowledge_base_path"] = str(kb_path)
    else:
        result["paths"]["knowledge_base_path"] = None

    # 출력 디렉토리 설정 (지역별 분리)
    output_dir = base_dir / "output" / region_config["name_en"]
    output_dir.mkdir(parents=True, exist_ok=True)
    result["paths"]["output_dir"] = str(output_dir)

    # CRS 정보
    result["paths"]["crs"] = region_config["crs"]

    # 지역명 정보 (보고서/플롯 제목에 사용)
    result["paths"]["region_name_kr"] = region_config["name_kr"]
    result["paths"]["region_name_en"] = region_config["name_en"]

    # Shapefile 이름 딕셔너리 (reconfigure에서 사용)
    result["paths"]["shapefiles"] = region_config["shapefiles"]

    # 최종 판정
    if not result["missing"]:
        result["available"] = True
        result["message"] = f"'{region}' 지역 데이터가 준비되었습니다."
    else:
        result["message"] = f"'{region}' 지역 데이터 중 일부가 누락되었습니다: {result['missing']}"

    return result


def get_available_regions() -> list:
    """사용 가능한 지역 목록 반환"""
    return list(REGIONS.keys())


def get_region_info(region: str) -> Optional[Dict[str, Any]]:
    """지역 정보 반환"""
    region = normalize_region_name(region)
    return REGIONS.get(region)
