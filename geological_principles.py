"""
Universal Geological Principles Module
범용 지질학 원리 모듈

지역별 지식베이스(KB)와 분리된, 범용 지질학적 원리를 정의.
AI 지질학자가 "(원리) + (지역 근거) → (결론)" 형태의 reasoning을 생성하도록 지원.

이 모듈은 수동 큐레이션되며, 외부 의존성 없음.
"""

# =============================================================================
# 1. BODY_GEOMETRY_PRINCIPLES — 관입체 형태별 생성 메커니즘 및 진단 기준
# =============================================================================

BODY_GEOMETRY_PRINCIPLES = {
    "dike": {
        "name_kr": "암맥",
        "name_en": "Dike",
        "mechanism": "마그마가 기존 지층의 절리·단층면을 따라 수직~고각으로 관입",
        "diagnostic_criteria": [
            "기존 지층을 가로지르는(부정합) 수직~급경사 판상 관입체",
            "폭 수 cm ~ 수십 m, 길이 수백 m ~ 수 km",
            "접촉부에 열변성(혼펠스) 발달 가능"
        ],
        "cross_section_shape": "수직~고각 판상 (narrow, steeply dipping tabular)",
        "typical_dip": "70-90°",
        "contact_relation": "부정합 (discordant)",
        "korea_applicable": True
    },
    "sill": {
        "name_kr": "암상",
        "name_en": "Sill",
        "mechanism": "마그마가 기존 지층의 층리면을 따라 수평으로 관입",
        "diagnostic_criteria": [
            "기존 지층과 평행한(정합) 수평~완경사 판상 관입체",
            "상·하 접촉부 모두에 열변성대 발달",
            "두께 수 m ~ 수백 m"
        ],
        "cross_section_shape": "수평 판상 (horizontal tabular)",
        "typical_dip": "0-15° (모암 층리면 경사와 동일)",
        "contact_relation": "정합 (concordant)",
        "korea_applicable": True
    },
    "laccolith": {
        "name_kr": "병반",
        "name_en": "Laccolith",
        "mechanism": "고점성 마그마가 지층 사이에 관입하여 상부 지층을 돔형으로 밀어올림",
        "diagnostic_criteria": [
            "하부: 평탄 (원래 층리면을 따라 관입한 면)",
            "상부: 돔형 볼록 (위의 지층을 밀어올린 형태)",
            "단면: 플라노-볼록 렌즈 / 버섯 갓 형태 (대칭 돔이 아님!)",
            "천부 관입 (얕은 심도)에서만 보존",
            "고점성 산성~중성 마그마에서 주로 형성"
        ],
        "cross_section_shape": "평탄 하부 + 돔상 상부 (plano-convex lens, mushroom cap)",
        "typical_dip": "하부 접촉면 0-15°, 상부 접촉면 20-60° (가장자리에서 급경사)",
        "contact_relation": "정합 (concordant) — 층리면 사이 관입",
        "korea_applicable": True,
        "condition": "천부 관입 지역에서만 보존 (심부 침식 지역에서는 이미 소멸)"
    },
    "lopolith": {
        "name_kr": "분반",
        "name_en": "Lopolith",
        "mechanism": "대규모 마그마가 안정 대륙지각에 쟁반형으로 관입, 자체 무게로 중심부 침강",
        "diagnostic_criteria": [
            "대규모 (수십~수백 km 직경)",
            "오목한 쟁반 형태 (saucer-shaped)",
            "안정 대륙 지각 환경 필요"
        ],
        "cross_section_shape": "오목한 쟁반 (saucer-shaped concave)",
        "typical_dip": "가장자리에서 중심으로 10-30° 경사",
        "contact_relation": "정합 (concordant)",
        "korea_applicable": False,
        "exclusion_reason": "안정적이고 넓은 대륙지각 필요 (남아공 Bushveld, 북미 Duluth 등). 한국은 지각 변동이 잦아 대규모 쟁반 형태 보존 불가."
    },
    "phacolith": {
        "name_kr": "습곡관입체",
        "name_en": "Phacolith",
        "mechanism": "습곡 구조의 힌지(배사 정점 또는 향사 저점)에 렌즈형 마그마 관입",
        "diagnostic_criteria": [
            "습곡 힌지에 위치",
            "렌즈형 단면",
            "안정적 습곡 구조 필요"
        ],
        "cross_section_shape": "습곡 힌지부 렌즈형 (lens-shaped at fold hinges)",
        "typical_dip": "습곡축 형태에 의존",
        "contact_relation": "정합 (concordant)",
        "korea_applicable": False,
        "exclusion_reason": "안정 대륙지각에서 대규모 습곡이 잘 보존된 환경 필요. 한국은 반복적 지구조 변동으로 보존 어려움."
    },
    "batholith": {
        "name_kr": "저반",
        "name_en": "Batholith",
        "mechanism": "대규모 마그마 방(magma chamber)이 서서히 냉각·고결, 장기간 관입 복합체",
        "diagnostic_criteria": [
            "노출 면적 100km² 이상",
            "부정합 접촉 (주변 암석을 관통)",
            "접촉부 외경사 (70-80°, 바깥으로 기울어짐)",
            "깊이에 따라 확장 (V자형 하방 확대)"
        ],
        "cross_section_shape": "대규모 V자 하방 확장 (downward-widening V-shape)",
        "typical_dip": "접촉면 70-80° 외경사",
        "contact_relation": "부정합 (discordant)",
        "korea_applicable": True
    },
    "stock": {
        "name_kr": "암주",
        "name_en": "Stock",
        "mechanism": "소규모 관입체, batholith의 일부가 노출되거나 독립적 소규모 관입",
        "diagnostic_criteria": [
            "노출 면적 100km² 미만",
            "부정합 접촉",
            "batholith보다 작지만 유사 구조"
        ],
        "cross_section_shape": "소규모 V자 확장 (smaller V-shape)",
        "typical_dip": "접촉면 65-80° 외경사",
        "contact_relation": "부정합 (discordant)",
        "korea_applicable": True
    },
    "volcanic_neck": {
        "name_kr": "화산암경",
        "name_en": "Volcanic Neck",
        "mechanism": "화산 분출 통로(vent)가 마그마로 채워진 후 주변부 침식으로 노출",
        "diagnostic_criteria": [
            "원형~타원형 평면 형태",
            "거의 수직인 급경사 접촉",
            "원통형 단면",
            "화산 최후기 산성암에서 흔함"
        ],
        "cross_section_shape": "수직 원통형 (vertical cylindrical)",
        "typical_dip": "80-90° (거의 수직)",
        "contact_relation": "부정합 (discordant)",
        "korea_applicable": True
    }
}


# =============================================================================
# 2. CONTACT_PRINCIPLES — 접촉 유형별 판정 규칙
# =============================================================================

CONTACT_PRINCIPLES = {
    "conformable": {
        "name_kr": "정합",
        "definition": "연속적 퇴적 또는 화산 활동으로 형성된 시간적 연속성이 있는 접촉",
        "diagnostic_rules": [
            "퇴적층이 시간 간격 없이 연속적으로 쌓인 경우",
            "화산 파편(각력암)과 용암이 짧은 시간 교대로 분출된 경우 (예: Kan-Kanb)",
            "같은 화산 활동 주기 내의 암석 조합",
            "층리면이 서로 평행하게 연장"
        ],
        "typical_dip_range": "모암 층리면 경사와 동일 (일반적으로 10-40°)",
        "contact_geometry": "평면~완만한 곡면"
    },
    "intrusive": {
        "name_kr": "관입",
        "definition": "마그마가 기존 암석을 뚫고 들어간 접촉. 열변성대 발달.",
        "diagnostic_rules": [
            "한쪽이 화성암이고 상대적으로 젊은 경우",
            "접촉부에 혼펠스(접촉변성암) 발달",
            "포획암(xenolith)이 관입암 내부에 존재",
            "냉각주변상(chilled margin)이 관입암 가장자리에 발달"
        ],
        "typical_dip_range": "형태에 따라 크게 다름 (dike: 70-90°, sill: 0-15°, batholith: 70-80°)",
        "contact_geometry": "형태에 따라 다양 (평면, 불규칙, 돔형)"
    },
    "unconformable": {
        "name_kr": "부정합",
        "definition": "퇴적 중단(침식면)을 사이에 둔 접촉. 시간 간극 존재.",
        "diagnostic_rules": [
            "하위 지층과 상위 지층 사이에 큰 시간 간극",
            "침식면(erosion surface)의 존재",
            "기저역암(basal conglomerate)이 부정합면 위에 발달 가능",
            "하위 지층이 절단되거나 풍화된 흔적"
        ],
        "typical_dip_range": "부정합면 자체는 0-15° (수평에 가까움), 하위 지층은 다양",
        "contact_geometry": "불규칙한 침식면 (irregular erosion surface)"
    },
    "fault": {
        "name_kr": "단층",
        "definition": "지각 응력에 의한 파쇄·변위면을 따른 접촉.",
        "diagnostic_rules": [
            "양쪽 암체의 변위(어긋남) 증거",
            "단층비지(fault gouge), 단층각력(fault breccia) 발달",
            "슬리켄사이드(slickenside, 마찰 줄무늬) 관찰",
            "선형적 지형 (lineament) 표현"
        ],
        "typical_dip_range": "정단층 50-70°, 역단층 30-50°, 주향이동단층 70-90°",
        "contact_geometry": "평면~리스트릭(곡면)"
    }
}


# =============================================================================
# 3. REGIONAL_INTERPRETATION_RULES — 지역 지질 역사에 따른 해석 규칙
# =============================================================================

REGIONAL_INTERPRETATION_RULES = {
    "서울": {
        "tectonic_setting": "선캠브리아 변성암 기반 + 쥬라기~백악기 심부 관입",
        "erosion_level": "심부 침식 (deep erosion)",
        "description": (
            "서울은 수억 년 동안 상부 지층이 깎여 나가면서, "
            "깊은 곳의 거대한 마그마 덩어리 '몸통'만 드러난 상태. "
            "얕은 곳에서 생기는 병반(laccolith) 구조는 이미 수백만 년 전에 침식으로 소멸."
        ),
        "body_geometry_overrides": {
            # 서울에서 고점성 산성암 → laccolith가 아니라 dike (관입 통로의 잔재)
            "high_viscosity_acidic": "dike",
            "default_intrusive": "stock",
        },
        "excluded_geometries": ["laccolith", "sill"],
        "exclusion_reason": "심부 침식 지역 — 천부 관입 형태(laccolith, sill)는 침식으로 소멸"
    },
    "부산": {
        "tectonic_setting": "경상분지 백악기 화산 활동 지역",
        "erosion_level": "천부 (shallow erosion)",
        "description": (
            "부산은 백악기 화산활동 지역으로 비교적 얕은 심도의 관입이 특징. "
            "상부 지층이 아직 보존되어 있어 laccolith 구조가 남아있을 수 있음. "
            "유천층군·하양층군 내 화산암들이 층상으로 반복 나타남."
        ),
        "body_geometry_overrides": {
            "high_viscosity_acidic": "laccolith",
            "default_intrusive": "stock",
        },
        "excluded_geometries": [],
        "exclusion_reason": None
    }
}


# =============================================================================
# 4. COMPOSITION_GEOMETRY_RULES — 암석 조성 → 형태 판정 규칙
# =============================================================================

COMPOSITION_GEOMETRY_RULES = [
    {
        "condition": "고점성 산성~중성 마그마 + 천부 관입 환경",
        "result": "laccolith",
        "reasoning": (
            "점성이 높은 마그마(유문암, 유문석영안산암 등)가 지표를 뚫지 못하고 "
            "지하 지층 사이에 고이면 laccolith를 형성. "
            "단, 천부 관입 지역(얕은 침식)에서만 보존됨."
        ),
        "applicable_compositions": ["rhyolite", "rhyodacite", "dacite", "유문암", "유문석영안산암"],
        "requires_shallow_erosion": True
    },
    {
        "condition": "고점성 산성~중성 마그마 + 심부 침식 환경",
        "result": "dike",
        "reasoning": (
            "같은 고점성 마그마라도 심부 침식 환경에서는 "
            "관입 통로의 잔재인 dike로만 관찰됨. "
            "원래 laccolith였더라도 상부가 침식으로 제거."
        ),
        "applicable_compositions": ["rhyolite", "rhyodacite", "dacite", "유문암", "유문석영안산암"],
        "requires_shallow_erosion": False
    },
    {
        "condition": "화산 파편암 + 동일 조성 용암의 교호",
        "result": "conformable (flow)",
        "reasoning": (
            "화산 파편(각력암)과 용암이 짧은 시간 교대로 분출된 것은 "
            "정합적인 화산 활동을 의미. 예: 안산암(Ka)과 안산암질화산각력암(Kavb)."
        ),
        "applicable_compositions": ["andesite+breccia", "안산암+화산각력암"],
        "requires_shallow_erosion": None
    },
    {
        "condition": "대규모 화강암 관입 (노출 면적 >100km²)",
        "result": "batholith",
        "reasoning": "대규모 마그마 방(magma chamber)이 장기간 관입·냉각한 복합체.",
        "applicable_compositions": ["granite", "granodiorite", "화강암", "화강섬록암"],
        "requires_shallow_erosion": None
    },
    {
        "condition": "소규모 관입 (~수 km²) + 부정합 접촉",
        "result": "stock",
        "reasoning": "batholith의 일부 노출이거나 독립적 소규모 관입.",
        "applicable_compositions": ["granite", "gabbro", "화강암", "반려암"],
        "requires_shallow_erosion": None
    },
    {
        "condition": "최후기 산성 화산암 + 원형~타원형 분포 + 급경사 접촉",
        "result": "volcanic_neck",
        "reasoning": "화산 분출 통로(vent)가 마그마로 채워진 후 주변부 침식으로 노출.",
        "applicable_compositions": ["rhyolite", "유문암", "유문암질암"],
        "requires_shallow_erosion": None
    }
]


# =============================================================================
# 5. KOREA_SPECIFIC_EXCLUSIONS — 한반도 제외 형태 + 근거
# =============================================================================

KOREA_SPECIFIC_EXCLUSIONS = {
    "lopolith": {
        "name_kr": "분반",
        "reason": (
            "대륙 지각이 매우 안정적이고 넓은 곳(남아공 Bushveld, 북미 Duluth 등)에서 "
            "대규모 마그마가 고여야 형성. 한국은 지각 변동이 잦아 "
            "대규모 쟁반 형태가 보존되기 어려움."
        ),
        "safe_fallback": "stock"
    },
    "phacolith": {
        "name_kr": "습곡관입체",
        "reason": (
            "안정 대륙지각에서 대규모 습곡이 잘 보존된 환경 필요. "
            "한국은 반복적 지구조 변동으로 습곡 힌지부의 렌즈형 관입이 보존되기 어려움."
        ),
        "safe_fallback": "stock"
    }
}


# =============================================================================
# Helper Functions
# =============================================================================

def generate_principles_context(rock_codes, region_name=None):
    """LLM 프롬프트에 삽입할 원리 컨텍스트 생성.

    rock_codes와 region_name으로 관련 원리만 필터링하여 반환.

    Args:
        rock_codes: list of LITHOIDX codes (e.g., ['Kan', 'Kbgr'])
        region_name: 지역명 (e.g., '부산', '서울') or None

    Returns:
        str: LLM 프롬프트에 삽입할 텍스트
    """
    sections = []

    # --- 1) 한반도 제외 형태 ---
    exclusion_lines = []
    for geo_type, info in KOREA_SPECIFIC_EXCLUSIONS.items():
        principle = BODY_GEOMETRY_PRINCIPLES.get(geo_type, {})
        exclusion_lines.append(
            f"- **{geo_type}** ({info['name_kr']}): 한반도 부적용. "
            f"사유: {info['reason']}"
        )
    if exclusion_lines:
        sections.append(
            "### 한반도 부적용 관입체 형태\n"
            "다음 형태는 한반도에서 관찰되지 않으므로 판정에서 제외:\n"
            + "\n".join(exclusion_lines)
        )

    # --- 2) 지역별 해석 규칙 ---
    if region_name and region_name in REGIONAL_INTERPRETATION_RULES:
        rules = REGIONAL_INTERPRETATION_RULES[region_name]
        region_section = f"### {region_name} 지역 해석 규칙\n"
        region_section += f"- 지구조 환경: {rules['tectonic_setting']}\n"
        region_section += f"- 침식 수준: {rules['erosion_level']}\n"
        region_section += f"- {rules['description']}\n"
        if rules['excluded_geometries']:
            region_section += (
                f"- 이 지역 제외 형태: {', '.join(rules['excluded_geometries'])} "
                f"({rules['exclusion_reason']})\n"
            )
        sections.append(region_section)

    # --- 3) 관련 접촉 원리 ---
    # 항상 주요 접촉 원리 포함 (LLM이 판정에 활용)
    contact_lines = []
    for ctype, info in CONTACT_PRINCIPLES.items():
        rules_text = "; ".join(info["diagnostic_rules"][:2])  # 상위 2개만
        contact_lines.append(
            f"- **{ctype}** ({info['name_kr']}): {info['definition']} "
            f"[판정: {rules_text}] "
            f"[경사: {info['typical_dip_range']}]"
        )
    sections.append(
        "### 접촉 유형 판정 원리\n" + "\n".join(contact_lines)
    )

    # --- 4) 관련 조성-형태 규칙 ---
    comp_lines = []
    for rule in COMPOSITION_GEOMETRY_RULES:
        comp_lines.append(
            f"- {rule['condition']} → **{rule['result']}**: {rule['reasoning']}"
        )
    sections.append(
        "### 암석 조성 → 관입체 형태 판정 원리\n" + "\n".join(comp_lines)
    )

    # --- 5) 관련 관입체 형태 원리 (적용 가능한 것만) ---
    applicable_types = [
        k for k, v in BODY_GEOMETRY_PRINCIPLES.items()
        if v.get("korea_applicable", True)
    ]
    geo_lines = []
    for geo_type in applicable_types:
        info = BODY_GEOMETRY_PRINCIPLES[geo_type]
        criteria = "; ".join(info["diagnostic_criteria"][:2])
        geo_lines.append(
            f"- **{geo_type}** ({info['name_kr']}): {info['mechanism']} "
            f"[진단: {criteria}] "
            f"[단면: {info['cross_section_shape']}]"
        )
    sections.append(
        "### 관입체 형태별 진단 기준 (한반도 적용 가능)\n" + "\n".join(geo_lines)
    )

    return "\n\n".join(sections)


def get_regional_body_geometry_override(region_name, rock_composition_hint=None):
    """지역별 해석 규칙에 따른 body_geometry 오버라이드.

    Args:
        region_name: 지역명 (e.g., '부산', '서울')
        rock_composition_hint: 암석 조성 힌트 (e.g., 'high_viscosity_acidic')

    Returns:
        str or None: 오버라이드할 body_geometry, 또는 None (오버라이드 불필요)
    """
    if not region_name or region_name not in REGIONAL_INTERPRETATION_RULES:
        return None

    rules = REGIONAL_INTERPRETATION_RULES[region_name]
    overrides = rules.get("body_geometry_overrides", {})

    # 조성 힌트가 있으면 해당 오버라이드 반환
    if rock_composition_hint and rock_composition_hint in overrides:
        return overrides[rock_composition_hint]

    return None
