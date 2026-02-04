"""
Seoul Geological Map Knowledge Base
서울 지질도폭 설명서 기반 지질 지식 베이스

Source: 지질도폭설명서 서울 1:50,000 (홍승호, 이병주, 황상기, 1982)
한국동력자원연구소
"""

# =============================================================================
# 암석 단위별 상세 정보
# =============================================================================

ROCK_UNITS = {
    "PCEbngn": {
        "name_kr": "호상흑운모편마암",
        "name_en": "Banded Biotite Gneiss",
        "age": "선캠브리아기 (Precambrian)",
        "age_range_ma": "800-2900",
        "description": """
이 도폭 전역에 걸쳐 가장 큰 암체인 이 편마암은 지형적으로 저지대를 이루면서
강서구 화곡동, 신정동, 서대문구 연희동, 신사동, 신도읍 향동리 등지에 넓게 분포한다.
노출은 불량한 편이다.
""",
        "lithology": {
            "texture": "호상구조 (Banded structure)",
            "grain_size": "조립질, 등립 변정질 (Granoblastic)",
            "composition": {
                "major": ["석영 (Quartz)", "사장석 (Plagioclase)", "흑운모 (Biotite)"],
                "minor": ["카리장석", "백운모", "녹니석", "견운모", "인회석", "저어콘", "규선석"]
            },
            "special_features": [
                "암색대(Melasome): 흑운모로 구성",
                "명색대(Leucosome): 석영, 장석으로 구성 - 화강암질 물질 주입",
                "노량진동, 대덕산 남부: 후생 석류석 포함"
            ]
        },
        "origin": "퇴적암 기원 준편마암 (Para-gneiss) - 석회암, 규암 협재",
        "metamorphic_grade": "각섬암상 (Amphibolite facies)",
        "structure": {
            "foliation_typical_dip_direction": "110-130° (SE)",
            "foliation_typical_dip_angle": "40-50°",
            "deformation_phases": "4차 이상의 변형작용으로 교란",
            "notes": "인접 지역 대비 반대 경사 방향 곳도 있어 습곡 구조 뚜렷"
        },
        "contact_characteristics": {
            "with_intrusive": "중생대 화성암류에 의해 관입됨",
            "internal": "석회암, 규암과 정합적 협재"
        },
        "weathering": "고기 변성암류로 훨씬 낮은 지형으로 풍화됨",
        "expected_dip_range": "40-70°",
        "expected_contact_type": "conformable"
    },

    "PCEls": {
        "name_kr": "결정질석회암",
        "name_en": "Crystalline Limestone",
        "age": "선캠브리아기 (Precambrian)",
        "description": """
김포군 고촌면 천둥고개 남북, 고양군 지도면 대상리, 부천시 도당동,
구로구 고척동과 신월동 일대에 반복 분포되어 구조 해석의 기준층으로 활용된다.
""",
        "lithology": {
            "texture": "당정질 석리 (Saccharoidal texture)",
            "color": "회색 내지 담록색",
            "composition": {
                "major": ["방해석 (Calcite)", "석영", "정장석"],
                "minor": ["각섬석", "휘석", "인회석", "녹니석", "스핀", "불투명 광물"]
            }
        },
        "associated_rocks": "석회규산염암 (투휘석 ~50%)",
        "occurrence": "호상흑운모편마암 내 협재",
        "foliation": "편마암의 편리 방향과 거의 일치",
        "contact_characteristics": {
            "with_gneiss": "정합적 협재, 편리 방향 일치"
        },
        "expected_dip_range": "40-70° (주변 편마암과 조화적)",
        "expected_contact_type": "conformable"
    },

    "PCEqz": {
        "name_kr": "규암",
        "name_en": "Quartzite",
        "age": "선캠브리아기 (Precambrian)",
        "description": """
부천시 여월동, 강서구 신정동, 목동, 고양군 지도면 강매리 등지에 소량 분포.
폭 약 2-3m, 연장 300m 미만. 강매리에서는 규암 주변부에 운모편암 동반.
""",
        "lithology": {
            "texture": "감합석리 (Mosaic texture)",
            "color": "담황색 내지 유백색",
            "grain_size": "세립질",
            "composition": {
                "major": ["석영 (~80%)"],
                "minor": ["백운모", "규선석"]
            },
            "special_features": ["파동소광", "봉합상 입자경계 (Sutured grain boundary)"]
        },
        "occurrence": "호상흑운모편마암 내 조화적 협재",
        "foliation": "편마암류의 편리방향과 거의 평행",
        "discontinuity_cause": "전위(Transposition)에 의한 이산 분포",
        "contact_characteristics": {
            "with_gneiss": "정합적 협재, 편리 평행"
        },
        "expected_dip_range": "40-70° (주변 편마암과 조화적)",
        "expected_contact_type": "conformable"
    },

    "PCEggn": {
        "name_kr": "화강암질편마암",
        "name_en": "Granitic Gneiss",
        "age": "선캠브리아기 (Precambrian)",
        "description": """
김포군 고촌면 신곡리, 부천시 여월동, 강서구 신정동, 내발산동,
서대문구 상암동, 고양군 지도면 용두리 등 일대에 분포.
재용융암(Anatexite)으로 해석됨.
""",
        "lithology": {
            "texture": "입상 변정질 조직 (Granoblastic texture)",
            "grain_size": "중립 내지 조립질",
            "composition": {
                "major": ["석영", "흑운모", "사장석"],
                "minor": ["석류석", "백운모", "견운모", "녹렴석", "규선석", "녹니석"]
            },
            "special_features": [
                "석류석 반상변정 (0.5-2mm, 최대 1cm 이상)",
                "취반상 변정 (Poikiloblastic texture)",
                "편암의 변성 잔류물(Relict) 포함"
            ]
        },
        "foliation": "희미한 엽리 또는 발달 불량",
        "contact_characteristics": {
            "with_banded_gneiss": "점이적(Gradational) 관계",
            "intruded_by": "후기 암맥류에 의해 관입"
        },
        "expected_dip_range": "40-70° (주변 편마암과 조화적일 때)",
        "expected_contact_type": "conformable"
    },

    "PCEam": {
        "name_kr": "각섬암",
        "name_en": "Amphibolite",
        "age": "선캠브리아기 (Precambrian)",
        "description": """
은평구 갈현동 서쪽과 고양군 원당면 도내리 북쪽에 소규모 분포.
고기의 염기성암이 변성작용을 받아 형성된 것으로 추정.
""",
        "lithology": {
            "color": "암녹색",
            "texture": "괴상 또는 부분적 엽리 발달",
            "composition": {
                "major": ["각섬석 (>60%)"],
                "minor": ["사장석", "흑운모", "정장석", "녹렴석", "불투명 광물"]
            }
        },
        "occurrence": "호상흑운모편마암과 조화적(Harmonious) 산출",
        "extent": "연장 약 200m 내외",
        "expected_dip_range": "40-70° (주변 편마암과 조화적)",
        "expected_contact_type": "conformable"
    },

    "Pgr": {
        "name_kr": "반상화강암",
        "name_en": "Porphyritic Granite",
        "age": "시대 미상 (흑운모화강암보다 선기)",
        "description": """
은평구 구파발동, 고양군 신도읍 삼송리 중심으로 암주(Stock)상 분포.
흑운모화강암보다 오래된 관입암체로 사료됨.
""",
        "lithology": {
            "texture": "반상조직 - 중립질 석기에 약 1cm 장석반정",
            "color": "백색 장석 (흑운모화강암의 담홍색과 구별됨)",
            "composition": {
                "major": ["석영", "정장석", "미사장석", "사장석", "흑운모"],
                "minor": ["녹니석", "인회석", "저어콘", "스핀", "불투명 광물"]
            },
            "special_features": ["문장석(Perthite) 발달", "동산리 부근 희미한 엽리"]
        },
        "contact_characteristics": {
            "with_gneiss": "관입 접촉",
            "with_biotite_granite": "직접 접촉부 없음"
        },
        "expected_dip_range": "70-90° (관입 접촉, 급경사)",
        "expected_contact_type": "intrusive"
    },

    "Jbgr": {
        "name_kr": "흑운모화강암",
        "name_en": "Jurassic Biotite Granite",
        "age": "쥬라기 (Jurassic)",
        "age_ma": "157-202",
        "description": """
북동부 인수봉에서 남산까지, 부천시 표절동, 작동, 강서구 개화동 일대 분포.
서울 화강암이라 불리는 대보화강암의 일부로, 대형 저반(Batholith)의 연장.
""",
        "lithology": {
            "texture": "등립상 조직 (Equigranular texture)",
            "grain_size": "중립 내지 조립질",
            "color": "담홍색 장석",
            "composition": {
                "major": ["석영", "사장석", "정장석", "미사장석", "흑운모"],
                "minor": ["철석", "백운모", "녹니석", "인회석", "스핀", "각섬석", "휘석"]
            },
            "special_features": [
                "문장석(Perthite) 발달",
                "연상석(Myrmekite) 형성",
                "누대구조 관찰",
                "박리도움(Exfoliation dome) 형성"
            ]
        },
        "emplacement": "매우 심처에서 서서히 냉각",
        "contact_characteristics": {
            "with_gneiss": "관입 접촉 - 안산 정상, 남산에서 관찰",
            "xenoliths": "북한리 근처에서 편마암 포획암 관찰"
        },
        "joints": {
            "set1": "055°-070° 방향, 60°-70° 경사",
            "set2": "155° 방향, 80° 경사"
        },
        "expected_dip_range": "70-90° (관입 접촉, 급경사)",
        "expected_contact_type": "intrusive"
    },

    "Kqv": {
        "name_kr": "석영맥",
        "name_en": "Quartz Vein",
        "age": "백악기 (Cretaceous)",
        "description": """
지도면 해매리, 원당면 용두리에서 폭 3-5m, 연장 약 200m 내외의 석영맥 분포.
변성암류 및 심성 관입암류를 관입.
""",
        "contact_characteristics": {
            "general": "관입 접촉, 거의 수직"
        },
        "expected_dip_range": "70-90° (암맥, 거의 수직)",
        "expected_contact_type": "intrusive"
    },

    "Kfl": {
        "name_kr": "규장암맥",
        "name_en": "Felsite Dike",
        "age": "백악기 (Cretaceous)",
        "description": "산성암맥류의 일종으로 도폭 내 도처에서 변성암류 및 심성 관입암류를 관입.",
        "expected_dip_range": "70-90° (암맥, 거의 수직)",
        "expected_contact_type": "intrusive"
    },

    "Kqp": {
        "name_kr": "석영반암",
        "name_en": "Quartz Porphyry",
        "age": "백악기 (Cretaceous)",
        "description": """
김포군 계양면 동양리 일대에 소규모 분포.
호상흑운모편마암을 관입하며, 약 0.5-1mm의 석영 반정을 가진다.
""",
        "lithology": {
            "texture": "반상조직 - 미정질 석기에 석영 반정",
            "color": "담회색 내지 담갈색",
            "composition": {
                "major": ["석영", "정장석"],
                "minor": ["사장석", "흑운모", "각섬석", "불투명 광물"]
            }
        },
        "expected_dip_range": "70-90° (암맥, 거의 수직)",
        "expected_contact_type": "intrusive"
    },

    "Qa": {
        "name_kr": "충적층",
        "name_en": "Alluvium",
        "age": "제4기 (Quaternary)",
        "description": """
한강 주변, 김포평야, 부평평야에 넓게 분포.
역, 사, 점토로 구성되며 미고결 상태.
부천시 오천동 일대 부평평야에 이탄(peat) 분포.
""",
        "lithology": {
            "composition": ["역 (Gravel)", "사 (Sand)", "점토 (Clay)"],
            "consolidation": "미고결"
        },
        "contact_characteristics": {
            "with_all_others": "부정합으로 피복"
        },
        "expected_dip_range": "0-10° (수평 내지 아수평)",
        "expected_contact_type": "unconformable"
    }
}


# =============================================================================
# 지질구조 정보
# =============================================================================

STRUCTURAL_GEOLOGY = {
    "regional_setting": {
        "location": "경기육괴 (Gyeonggi Massif) 중앙부",
        "dominant_trend": "NE-SW",
        "tectonic_history": "선캠브리아기 이래 4차에 걸친 습곡작용"
    },

    "foliation": {
        "S1_gneissosity": {
            "typical_dip_direction": "110-130° (SE)",
            "typical_dip_angle": "40-50°",
            "variation": "4차 변형작용으로 곳에 따라 다양",
            "han_river_north": "분산도 높음",
            "han_river_south": "상대적으로 일관성"
        },
        "S2_secondary_foliation": {
            "description": "밀접(Tight) 습곡축면",
            "relation_to_S1": "평행하거나 10° 미만 사교",
            "dip_direction": "010-030°"
        }
    },

    "folding": {
        "F1": {
            "type": "엽리내 습곡 (Intrafolial fold)",
            "characteristics": "변성작용과 수반, 편마암 구조에 평행"
        },
        "F2": {
            "products": ["광물배열 선구조", "밀접습곡"],
            "axis_trend": "010-200°"
        },
        "F3": {
            "type": "접근 습곡 (Close fold)",
            "characteristics": "가장 강한 습곡작용",
            "axis_trend": "010-030°",
            "significance": "S1면을 심하게 교란"
        },
        "F4": {
            "type": "개방 습곡 (Open fold)",
            "axis_trend": "ESE",
            "characteristics": "최후기 습곡작용, F1-F3 간섭"
        }
    },

    "faulting": {
        "han_river_north": {
            "dominant_direction": "N-S (거의 남북방향)",
            "other": "NE-E (성산동 남측)"
        },
        "han_river_south": {
            "directions": ["NE-E", "NW-W"]
        },
        "types": {
            "신정동_남측": "역단층",
            "남쪽": "정단층",
            "others": "주향이동 단층"
        },
        "main_fault": "한강을 따른 NW-W 방향 주단층"
    },

    "lineaments": {
        "dominant": "N-S (남북방향)",
        "secondary": "N30-40°E"
    },

    "joints_in_granite": {
        "set1": {"direction": "055-070°", "dip": "60-70°"},
        "set2": {"direction": "155°", "dip": "~80°"},
        "stress_field": "δ1 거의 동서방향"
    }
}


# =============================================================================
# 접촉 관계 규칙
# =============================================================================

CONTACT_RULES = {
    "conformable_pairs": [
        ("PCEbngn", "PCEls"),   # 편마암-석회암: 협재
        ("PCEbngn", "PCEqz"),   # 편마암-규암: 협재
        ("PCEbngn", "PCEggn"),  # 편마암-화강암질편마암: 점이적
        ("PCEbngn", "PCEam"),   # 편마암-각섬암: 조화적
        ("PCEls", "PCEqz"),     # 석회암-규암: 모두 편마암 내 협재
        ("PCEggn", "PCEam"),    # 화강암질편마암-각섬암
    ],

    "intrusive_pairs": [
        ("PCEbngn", "Pgr"),     # 편마암 ← 반상화강암 관입
        ("PCEbngn", "Jbgr"),    # 편마암 ← 흑운모화강암 관입
        ("PCEggn", "Jbgr"),     # 화강암질편마암 ← 흑운모화강암 관입
        ("PCEbngn", "Kqv"),     # 편마암 ← 석영맥 관입
        ("PCEbngn", "Kfl"),     # 편마암 ← 규장암맥 관입
        ("PCEbngn", "Kqp"),     # 편마암 ← 석영반암 관입
        ("PCEggn", "Kqv"),      # 화강암질편마암 ← 석영맥 관입
        ("Jbgr", "Kqv"),        # 흑운모화강암 ← 석영맥 관입
        ("Jbgr", "Kfl"),        # 흑운모화강암 ← 규장암맥 관입
    ],

    "unconformable_pairs": [
        ("PCEbngn", "Qa"),      # 모든 암석 위 충적층 부정합
        ("PCEggn", "Qa"),
        ("PCEls", "Qa"),
        ("Pgr", "Qa"),
        ("Jbgr", "Qa"),
        ("Kqv", "Qa"),
        ("Kfl", "Qa"),
    ]
}


# =============================================================================
# 경사각 추정 규칙
# =============================================================================

DIP_ESTIMATION_RULES = {
    "precambrian_metamorphic": {
        "typical_range": "40-70°",
        "dominant_direction": "SE (110-130°)",
        "notes": "4차 변형작용으로 국지적 변이 존재"
    },

    "mesozoic_intrusive": {
        "typical_range": "70-90°",
        "notes": "관입 접촉은 일반적으로 급경사"
    },

    "cretaceous_dikes": {
        "typical_range": "70-90° (거의 수직)",
        "notes": "암맥은 일반적으로 급경사"
    },

    "quaternary_alluvium": {
        "typical_range": "0-10°",
        "notes": "부정합으로 기반암 피복, 수평에 가까움"
    },

    "contact_type_rules": {
        "conformable": {
            "description": "정합적 접촉 (층상, 협재)",
            "dip_follows": "주변 엽리 방향과 일치",
            "typical_angle": "40-70°"
        },
        "intrusive": {
            "description": "관입 접촉",
            "characteristics": "급경사, 엽리 절단 가능",
            "typical_angle": "70-90°"
        },
        "unconformable": {
            "description": "부정합 접촉",
            "characteristics": "하부 구조 절단, 수평에 가까움",
            "typical_angle": "0-15°"
        }
    }
}


# =============================================================================
# 암석쌍별 접촉 유형 결정 함수
# =============================================================================

def get_contact_type(rock1: str, rock2: str) -> str:
    """
    두 암석 사이의 접촉 유형을 반환합니다.

    Returns:
        "conformable", "intrusive", "unconformable", or "unknown"
    """
    pair = tuple(sorted([rock1, rock2]))

    # Check conformable
    for p in CONTACT_RULES["conformable_pairs"]:
        if tuple(sorted(p)) == pair:
            return "conformable"

    # Check intrusive
    for p in CONTACT_RULES["intrusive_pairs"]:
        if tuple(sorted(p)) == pair:
            return "intrusive"

    # Check unconformable
    for p in CONTACT_RULES["unconformable_pairs"]:
        if tuple(sorted(p)) == pair:
            return "unconformable"

    return "unknown"


def get_expected_dip_range(rock1: str, rock2: str) -> dict:
    """
    두 암석 사이의 예상 경사각 범위를 반환합니다.
    """
    contact_type = get_contact_type(rock1, rock2)

    if contact_type == "unconformable":
        return {"min": 0, "max": 15, "typical": 5}
    elif contact_type == "intrusive":
        return {"min": 70, "max": 90, "typical": 75}
    elif contact_type == "conformable":
        return {"min": 40, "max": 70, "typical": 55}
    else:
        # Unknown - return wide range
        return {"min": 20, "max": 80, "typical": 50}


def get_rock_description(lithoidx: str) -> dict:
    """
    암석 코드에 대한 상세 정보를 반환합니다.
    """
    return ROCK_UNITS.get(lithoidx, {
        "name_kr": "알 수 없음",
        "name_en": "Unknown",
        "description": "정보 없음"
    })


# =============================================================================
# 지질 컨텍스트 문자열 생성 함수
# =============================================================================

def generate_context_for_boundary(rock_codes: list) -> str:
    """
    주어진 암석들에 대한 지질학적 컨텍스트를 생성합니다.
    """
    context_parts = []

    context_parts.append("## 서울 지역 지질도폭 설명서 기반 지식\n")

    # Rock descriptions
    context_parts.append("### 관련 암석 단위\n")
    for code in rock_codes:
        if code in ROCK_UNITS:
            unit = ROCK_UNITS[code]
            context_parts.append(f"**{code} ({unit['name_kr']}, {unit['name_en']})**")
            context_parts.append(f"- 시대: {unit.get('age', '불명')}")
            if 'expected_dip_range' in unit:
                context_parts.append(f"- 예상 경사: {unit['expected_dip_range']}")
            if 'expected_contact_type' in unit:
                context_parts.append(f"- 일반적 접촉유형: {unit['expected_contact_type']}")
            context_parts.append("")

    # Contact relationships
    if len(rock_codes) >= 2:
        context_parts.append("### 접촉 관계 분석\n")
        for i, r1 in enumerate(rock_codes):
            for r2 in rock_codes[i+1:]:
                ct = get_contact_type(r1, r2)
                dip_range = get_expected_dip_range(r1, r2)
                context_parts.append(f"- {r1}-{r2}: {ct} 접촉")
                context_parts.append(f"  예상 경사: {dip_range['min']}°-{dip_range['max']}° (전형: {dip_range['typical']}°)")

    # Regional structure
    context_parts.append("\n### 지역 구조 특성")
    context_parts.append("- 편마구조 우세 경사방향: SE (110-130°)")
    context_parts.append("- 편마구조 우세 경사각: 40-50°")
    context_parts.append("- 주요 습곡축 방향: NE-SW (010-030°)")
    context_parts.append("- 4차에 걸친 변형작용으로 국지적 변이 존재")

    return '\n'.join(context_parts)


# =============================================================================
# 지역 컨텍스트 생성 (Regional Context Generation)
# =============================================================================

def generate_regional_context() -> str:
    """
    지식베이스에서 서울 지역 전체 지질 컨텍스트를 동적으로 생성합니다.
    """
    lines = []

    # Header
    lines.append("# 서울 지역 지질 개요 (Seoul Area Geological Context)")
    lines.append("# 출처: 지질도폭설명서 서울 1:50,000 (홍승호, 이병주, 황상기, 1982)")
    lines.append("")

    # Regional Setting
    regional = STRUCTURAL_GEOLOGY['regional_setting']
    lines.append("## 지역 개요 (Regional Setting)")
    lines.append(f"- 위치: {regional['location']}")
    lines.append(f"- 주요 구조 방향: {regional['dominant_trend']}")
    lines.append("- 주요 암석: 선캠브리아기 변성암류를 관입한 중생대 화강암류")
    lines.append("")

    # Rock Units - Precambrian
    lines.append("## 암석 단위 (Rock Units)")
    lines.append("")
    lines.append("### 선캠브리아기 변성암 (Precambrian Metamorphic Rocks)")

    precambrian_codes = ['PCEbngn', 'PCEggn', 'PCEls', 'PCEqz', 'PCEam']
    for code in precambrian_codes:
        if code in ROCK_UNITS:
            unit = ROCK_UNITS[code]
            lines.append(f"- **{code}** ({unit['name_kr']}): {unit.get('expected_dip_range', '40-70°')} 경사")
            if 'structure' in unit and 'foliation_typical_dip_direction' in unit['structure']:
                lines.append(f"  - 엽리: {unit['structure']['foliation_typical_dip_direction']}, 경사 {unit['structure']['foliation_typical_dip_angle']} 우세")
    lines.append("")

    # Rock Units - Mesozoic Intrusive
    lines.append("### 중생대 관입암 (Mesozoic Intrusive Rocks)")
    mesozoic_codes = ['Pgr', 'Jbgr']
    for code in mesozoic_codes:
        if code in ROCK_UNITS:
            unit = ROCK_UNITS[code]
            lines.append(f"- **{code}** ({unit['name_kr']}): {unit.get('expected_contact_dip', '70-90°')} 관입 접촉")
    lines.append("")

    # Rock Units - Cretaceous Dikes
    lines.append("### 백악기 암맥 (Cretaceous Dikes)")
    dike_codes = ['Kqp', 'Kqv', 'Kfl']
    for code in dike_codes:
        if code in ROCK_UNITS:
            unit = ROCK_UNITS[code]
            lines.append(f"- **{code}** ({unit['name_kr']}): 거의 수직 (70-90°)")
    lines.append("")

    # Rock Units - Quaternary
    lines.append("### 제4기 (Quaternary)")
    if 'Qa' in ROCK_UNITS:
        unit = ROCK_UNITS['Qa']
        lines.append(f"- **Qa** ({unit['name_kr']}): 수평~아수평 (<10°)")
    lines.append("")

    # Structural Geology
    lines.append("## 지질구조 (Structural Geology)")
    lines.append("")

    # Foliation
    lines.append("### 편마구조 (Gneissosity S1)")
    fol = STRUCTURAL_GEOLOGY['foliation']['S1_gneissosity']
    lines.append(f"- 우세 경사방향: {fol['typical_dip_direction']}")
    lines.append(f"- 우세 경사각: {fol['typical_dip_angle']}")
    lines.append(f"- 특이사항: {fol['variation']}")
    lines.append("")

    # Folding
    lines.append("### 습곡 작용 (4 Phases)")
    for phase, info in STRUCTURAL_GEOLOGY['folding'].items():
        if 'type' in info:
            lines.append(f"- {phase}: {info['type']}")
        if 'axis_trend' in info:
            lines.append(f"  축 방향: {info['axis_trend']}")
    lines.append("")

    # Faulting
    lines.append("### 단층")
    faulting = STRUCTURAL_GEOLOGY['faulting']
    lines.append(f"- 한강 북부: {faulting['han_river_north']['dominant_direction']} 방향 우세")
    lines.append(f"- 한강 남부: {', '.join(faulting['han_river_south']['directions'])} 방향")
    lines.append(f"- 주단층: {faulting['main_fault']}")
    lines.append("")

    # Contact Rules
    lines.append("## 접촉 관계 규칙 (Contact Relationship Rules)")
    lines.append("")

    rules = DIP_ESTIMATION_RULES['contact_type_rules']
    lines.append(f"1. **정합적 접촉** (Conformable): {rules['conformable']['dip_follows']}")
    lines.append(f"   - 전형 경사: {rules['conformable']['typical_angle']}")
    lines.append("")
    lines.append(f"2. **관입 접촉** (Intrusive): {rules['intrusive']['characteristics']}")
    lines.append(f"   - 전형 경사: {rules['intrusive']['typical_angle']}")
    lines.append("")
    lines.append(f"3. **부정합 접촉** (Unconformable): {rules['unconformable']['characteristics']}")
    lines.append(f"   - 전형 경사: {rules['unconformable']['typical_angle']}")
    lines.append("")

    # Dip Estimation Principles
    lines.append("## 경사 추정 원칙")
    lines.append("")
    lines.append("1. **3점법(Three-Point Method)**: 지형 횡단 경계에서 3점 이상 표고차로 주향/경사 계산")
    lines.append("")
    lines.append("2. **V-규칙(V-Rule)**: 계곡 횡단 시")
    lines.append("   - V가 상류 → 경사 하류 방향")
    lines.append("   - V가 하류 → 경사 상류 방향")
    lines.append("   - 직선 → 수직 접촉")
    lines.append("")
    lines.append("3. **노두폭 규칙(Contact Width Rule)**:")
    lines.append("   - 넓은 노두폭 → 완경사")
    lines.append("   - 좁은 노두폭 → 급경사")

    return '\n'.join(lines)


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    # Test contact type detection
    print("=== Contact Type Tests ===")
    print(f"PCEbngn-PCEls: {get_contact_type('PCEbngn', 'PCEls')}")
    print(f"PCEbngn-Jbgr: {get_contact_type('PCEbngn', 'Jbgr')}")
    print(f"Jbgr-Qa: {get_contact_type('Jbgr', 'Qa')}")

    # Test context generation
    print("\n=== Context Generation Test ===")
    context = generate_context_for_boundary(['PCEbngn', 'Jbgr', 'Qa'])
    print(context)
