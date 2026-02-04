"""
Geology Cross-Section Review Agent
지질 단면도 검토 LLM 에이전트

단면도 이미지를 분석하여 지질학적 타당성을 검토하고 개선점을 제안합니다.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import base64
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import anthropic

# Load configuration (supports CLI arguments: --config, --data-dir, --output-dir, --dem-file)
from config_loader import init_config

CONFIG = init_config()
PATHS = CONFIG['paths']

# Paths from config
CONFIG_FILE = PATHS['base_dir'] / "config.yaml"
OUTPUT_DIR = PATHS['output_dir']


# =============================================================================
# Review Agent Class
# =============================================================================

class GeologyReviewAgent:
    """LLM-based geological cross-section review agent"""

    def __init__(self, config_path: Path = None):
        """Initialize with Anthropic API"""
        config_path = config_path or CONFIG_FILE

        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            self.api_key = config.get('anthropic', {}).get('api_key')
            # 검토에는 더 강력한 모델 사용 권장
            self.model = config.get('model', {}).get('review_model', 'claude-sonnet-4-20250514')
            self.max_tokens = 4096  # 상세한 검토를 위해 더 많은 토큰

            # 검토 페르소나 로드
            self.system_prompt = config.get('review_persona', {}).get('system_prompt', '')
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if not self.api_key or self.api_key == "YOUR_API_KEY_HERE":
            raise ValueError(f"API 키를 config.yaml에 입력해주세요")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.reviews = []

        print(f"  Review Model: {self.model}")
        print(f"  Review Persona: {'Loaded from config' if self.system_prompt else 'Not configured'}")

    def _load_image_as_base64(self, image_path: Path) -> str:
        """Load image file and convert to base64"""
        with open(image_path, 'rb') as f:
            return base64.standard_b64encode(f.read()).decode('utf-8')

    def _create_review_prompt(self, section_name: str, section_info: Dict = None) -> str:
        """Create prompt for cross-section review"""

        prompt = f"""# 지질 단면도 검토 요청

## 단면 정보
- 단면명: {section_name}
"""
        if section_info:
            prompt += f"""- 시점: ({section_info.get('start', {}).get('x', '?')}, {section_info.get('start', {}).get('y', '?')})
- 종점: ({section_info.get('end', {}).get('x', '?')}, {section_info.get('end', {}).get('y', '?')})
- 방위각: {section_info.get('azimuth', '?')}°
"""

        prompt += """
## 암석 단위별 색상 범례 (Color Legend) - 반드시 참고하세요!

이미지에서 각 암석은 다음 색상으로 표시됩니다:

| 암석 코드 | 한글명 | 색상 | RGB 근사값 |
|-----------|--------|------|------------|
| **Qa** | 충적층 | **밝은 노란색 (순수 노랑)** | #FFFF00 |
| **PCEbngn** | 호상흑운모편마암 | **연분홍색** | #FFB6C1 |
| **PCEggn** | 화강편마암 | **연보라색 (자주색)** | #DDA0DD |
| **PCEls** | 결정질석회암 | **하늘색** | #87CEEB |
| **PCEam** | 각섬암 | **연두색** | #90EE90 |
| **Jbgr** | 흑운모화강암 | **토마토색 (주황-빨강)** | #FF6347 |
| **Pgr** | 반상화강암 | **연한 주황색** | #FFA07A |
| **Kfl** | 규장암 | **갈색-분홍** | #BC8F8F |

**중요 - 반드시 숙지하세요**:
- **충적층(Qa)은 오직 밝은 노란색(순수 노랑, #FFFF00)만** 해당합니다!
- 자주색/보라색(PCEggn), 분홍색(PCEbngn), 주황색(Jbgr) 등은 충적층이 **아닙니다**!
- 충적층은 지표면 근처에 렌즈형으로 얇게(10-30m) 표시되어 있습니다.

**회색/빈 공간에 대한 설명**:
- 지하 깊은 곳의 **회색 영역(#D3D3D3)**은 **미분화 기반암(undifferentiated basement)**입니다.
- 이 회색 영역은 **충적층이 아닙니다**! 지하 심부의 불확실한 지질을 나타냅니다.
- 암석 단위들 사이의 빈 공간/회색 영역은 "모르는 영역"이지 충적층이 침투한 것이 아닙니다.
- 충적층은 **오직 지표면에서만** 연한 노란색으로 표시됩니다.

## 검토 요청사항

위 이미지는 AI가 생성한 지질 단면도입니다.
당신의 전문 지식과 경험을 바탕으로 이 단면도를 **엄격하고 비판적으로** 검토해주세요.

**색상을 정확히 구분하여** 다음 사항을 중점 확인하세요:
1. **충적층(Qa, 연한 노란색/레몬색만!)**: 지표면을 따라 렌즈형으로 얇게 덮여 있는가?
   - ⚠️ **주의**: 지하의 회색 영역은 충적층이 아닙니다! 회색은 미분화 기반암입니다.
   - ⚠️ **주의**: 충적층이 "지하 깊이 침투"했다고 판단하기 전에, 그 영역이 정말 연한 노란색인지 확인하세요.
2. **관입체 형태 (Jbgr=토마토색, Pgr=주황색)**: 현실적인 저반/암주 형태인가? 완벽한 수직벽으로 되어있지 않은가?
   - 단면도에 수직과장(vertical exaggeration)이 있어 실제보다 가파르게 보일 수 있습니다.
3. **변성암 (PCEbngn=분홍색, PCEggn=자주색)**: 엽리 방향이 일관적인가?
4. **접촉 관계**: 층서적 원리에 맞는가?
5. **미분화 기반암 (회색 영역)**: 지하 깊은 곳의 불확실한 지질을 나타내며, 이는 정상적인 표현입니다.

다음 JSON 형식으로 응답해주세요:

```json
{{
    "overall_assessment": "<전체 평가: excellent/good/acceptable/poor/unacceptable>",
    "overall_score": <1-10 점수>,

    "critical_issues": [
        {{
            "issue": "<심각한 문제 설명>",
            "location": "<문제 위치 - 예: 거리 2000-3000m 구간>",
            "severity": "<critical/major/minor>",
            "geological_principle": "<위반된 지질학 원리>"
        }}
    ],

    "alluvium_review": {{
        "is_correct": <true/false>,
        "correctly_identified_yellow_only": <true/false>,
        "gray_areas_confused_with_alluvium": <true/false>,
        "issues": ["<충적층 표현 문제점들 - 연한 노란색 영역만 해당>"],
        "suggestions": ["<개선 제안>"],
        "note": "<회색 영역은 미분화 기반암이며 충적층이 아님을 인지했는지>"
    }},

    "intrusion_review": {{
        "shapes_realistic": <true/false>,
        "depth_behavior_correct": <true/false>,
        "issues": ["<관입체 형태 문제점>"],
        "suggestions": ["<개선 제안>"]
    }},

    "contact_relationships": {{
        "correct_relationships": ["<올바른 접촉 관계>"],
        "incorrect_relationships": ["<잘못된 접촉 관계>"],
        "suggestions": ["<개선 제안>"]
    }},

    "dip_angles": {{
        "realistic": <true/false>,
        "issues": ["<경사각 문제점>"],
        "suggestions": ["<개선 제안>"]
    }},

    "visual_quality": {{
        "clarity": "<clear/acceptable/confusing>",
        "color_scheme": "<appropriate/needs_improvement>",
        "labeling": "<adequate/insufficient>",
        "suggestions": ["<시각적 개선 제안>"]
    }},

    "geological_realism": {{
        "score": <1-10>,
        "strengths": ["<잘 표현된 점>"],
        "weaknesses": ["<개선 필요한 점>"]
    }},

    "specific_recommendations": [
        {{
            "priority": <1-5, 1이 가장 높음>,
            "recommendation": "<구체적 개선 권고>",
            "expected_impact": "<개선 시 기대 효과>"
        }}
    ],

    "summary_korean": "<전체 검토 요약 (한국어, 3-5문장)>"
}}
```

이미지를 면밀히 분석하고, 지질학적 타당성을 중심으로 검토해주세요.
특히 **충적층(Qa, 연한 노란색)이 지표면을 따라 올바르게 표현되었는지** 주의 깊게 확인해주세요.
"""

        return prompt

    def review_section(self, image_path: Path, section_name: str,
                       section_info: Dict = None) -> Optional[Dict]:
        """Review a single cross-section image"""

        if not image_path.exists():
            print(f"    Image not found: {image_path}")
            return None

        # Load image
        image_data = self._load_image_as_base64(image_path)

        # Determine media type
        suffix = image_path.suffix.lower()
        media_type = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }.get(suffix, 'image/png')

        # Create prompt
        prompt = self._create_review_prompt(section_name, section_info)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            response_text = response.content[0].text.strip()

            # Parse JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            result = json.loads(response_text)
            result['section_name'] = section_name
            result['image_path'] = str(image_path)
            result['review_timestamp'] = datetime.now().isoformat()

            return result

        except json.JSONDecodeError as e:
            print(f"    JSON parse error: {e}")
            print(f"    Response: {response_text[:500]}...")
            return None
        except Exception as e:
            print(f"    Error reviewing: {e}")
            return None

    def review_all_sections(self, section_dir: Path = None) -> List[Dict]:
        """Review all cross-section images in directory"""

        section_dir = section_dir or OUTPUT_DIR

        # Find all section images
        image_files = list(section_dir.glob("*_llm_section.png"))

        if not image_files:
            print("No cross-section images found.")
            return []

        print(f"\nFound {len(image_files)} cross-section images to review")
        print("-" * 50)

        results = []
        for i, img_path in enumerate(image_files):
            section_name = img_path.stem.replace("_llm_section", "")
            print(f"\n[{i+1}/{len(image_files)}] Reviewing: {section_name}...")

            result = self.review_section(img_path, section_name)

            if result:
                score = result.get('overall_score', 0)
                assessment = result.get('overall_assessment', 'unknown')
                critical = len(result.get('critical_issues', []))

                print(f"    Score: {score}/10 ({assessment})")
                print(f"    Critical issues: {critical}")

                results.append(result)
            else:
                print("    Review failed")

        self.reviews = results
        return results

    def generate_report(self) -> str:
        """Generate comprehensive review report"""

        if not self.reviews:
            return "No reviews available."

        lines = []
        lines.append("=" * 70)
        lines.append("  GEOLOGY CROSS-SECTION REVIEW REPORT")
        lines.append("  지질 단면도 검토 보고서")
        lines.append("=" * 70)
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"Sections reviewed: {len(self.reviews)}")

        # Overall statistics
        scores = [r.get('overall_score', 0) for r in self.reviews]
        lines.append(f"\n평균 점수: {sum(scores)/len(scores):.1f}/10")

        # Assessment distribution
        assessments = [r.get('overall_assessment', 'unknown') for r in self.reviews]
        lines.append("\n전체 평가 분포:")
        for a in set(assessments):
            count = assessments.count(a)
            lines.append(f"  {a}: {count}")

        # Critical issues summary
        all_critical = []
        for r in self.reviews:
            all_critical.extend(r.get('critical_issues', []))

        if all_critical:
            lines.append(f"\n{'='*70}")
            lines.append("중요 문제점 요약 (Critical Issues)")
            lines.append("="*70)
            for issue in all_critical:
                lines.append(f"\n[{issue.get('severity', '?').upper()}] {issue.get('issue', '')}")
                lines.append(f"  위치: {issue.get('location', '?')}")
                lines.append(f"  원리: {issue.get('geological_principle', '?')}")

        # Per-section details
        lines.append(f"\n{'='*70}")
        lines.append("단면별 상세 검토")
        lines.append("="*70)

        for r in self.reviews:
            lines.append(f"\n--- {r.get('section_name', 'Unknown')} ---")
            lines.append(f"점수: {r.get('overall_score', 0)}/10 ({r.get('overall_assessment', '?')})")
            lines.append(f"\n요약: {r.get('summary_korean', '')}")

            # Alluvium review
            alluvium = r.get('alluvium_review', {})
            if not alluvium.get('is_correct', True):
                lines.append(f"\n[충적층 문제]")
                for issue in alluvium.get('issues', []):
                    lines.append(f"  - {issue}")

            # Recommendations
            recs = r.get('specific_recommendations', [])
            if recs:
                lines.append(f"\n[개선 권고사항]")
                for rec in sorted(recs, key=lambda x: x.get('priority', 99)):
                    lines.append(f"  [{rec.get('priority', '?')}] {rec.get('recommendation', '')}")

        return '\n'.join(lines)

    def export_results(self, output_path: Path = None):
        """Export review results to JSON"""

        output_path = output_path or (OUTPUT_DIR / "review_results.json")

        output_data = {
            'review_summary': {
                'total_reviewed': len(self.reviews),
                'average_score': sum(r.get('overall_score', 0) for r in self.reviews) / len(self.reviews) if self.reviews else 0,
                'model_used': self.model,
                'timestamp': datetime.now().isoformat()
            },
            'reviews': self.reviews
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\nReview results exported to: {output_path}")
        return output_path

    def generate_html_report(self, output_path: Path = None) -> Path:
        """Generate comprehensive HTML review report"""

        output_path = output_path or (OUTPUT_DIR / "review_report.html")

        if not self.reviews:
            return None

        # Statistics
        scores = [r.get('overall_score', 0) for r in self.reviews]
        avg_score = sum(scores) / len(scores) if scores else 0
        assessments = [r.get('overall_assessment', 'unknown') for r in self.reviews]

        # Collect all critical issues
        all_critical = []
        for r in self.reviews:
            for issue in r.get('critical_issues', []):
                issue['section'] = r.get('section_name', 'Unknown')
                all_critical.append(issue)

        # Score color function
        def score_color(score):
            if score >= 7: return '#27ae60'
            elif score >= 5: return '#f39c12'
            else: return '#e74c3c'

        def severity_color(severity):
            return {'critical': '#e74c3c', 'major': '#f39c12', 'minor': '#3498db'}.get(severity, '#666')

        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Geology Cross-Section Review Report</title>
    <style>
        :root {{
            --primary: #2c3e50;
            --secondary: #3498db;
            --success: #27ae60;
            --warning: #f39c12;
            --danger: #e74c3c;
            --light: #ecf0f1;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Malgun Gothic', 'Segoe UI', sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
        }}
        header h1 {{
            color: var(--primary);
            font-size: 2em;
            margin-bottom: 10px;
        }}
        .score-overview {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        .score-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            padding: 25px 40px;
            text-align: center;
            min-width: 150px;
        }}
        .score-card .big-number {{
            font-size: 3em;
            font-weight: bold;
            color: {score_color(avg_score)};
        }}
        .score-card .label {{
            color: #666;
            font-size: 0.9em;
        }}
        .section {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: var(--primary);
            border-bottom: 3px solid var(--secondary);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .critical-issue {{
            background: #fff5f5;
            border-left: 4px solid var(--danger);
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 10px 10px 0;
        }}
        .major-issue {{
            background: #fffbf0;
            border-left: 4px solid var(--warning);
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 10px 10px 0;
        }}
        .minor-issue {{
            background: #f0f8ff;
            border-left: 4px solid var(--secondary);
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 10px 10px 0;
        }}
        .issue-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }}
        .severity-badge {{
            padding: 3px 10px;
            border-radius: 15px;
            color: white;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .section-badge {{
            background: #eee;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 0.8em;
        }}
        .review-card {{
            border: 1px solid #eee;
            border-radius: 10px;
            margin: 15px 0;
            overflow: hidden;
        }}
        .review-header {{
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .review-header h3 {{
            margin: 0;
        }}
        .review-score {{
            background: white;
            color: var(--primary);
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .review-body {{
            padding: 20px;
        }}
        .review-summary {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            font-style: italic;
        }}
        .sub-section {{
            margin: 15px 0;
        }}
        .sub-section h4 {{
            color: var(--secondary);
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .recommendation {{
            background: #e8f5e9;
            border-left: 4px solid var(--success);
            padding: 12px 15px;
            margin: 8px 0;
            border-radius: 0 8px 8px 0;
        }}
        .priority-badge {{
            background: var(--success);
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.8em;
            margin-right: 10px;
        }}
        .image-preview {{
            text-align: center;
            margin: 15px 0;
        }}
        .image-preview img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: transform 0.2s;
        }}
        .image-preview img:hover {{
            transform: scale(1.02);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-item .value {{
            font-size: 1.5em;
            font-weight: bold;
            color: var(--primary);
        }}
        footer {{
            text-align: center;
            padding: 20px;
            color: white;
            font-size: 0.9em;
        }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 9999;
            left: 0; top: 0;
            width: 100%; height: 100%;
            background: rgba(0,0,0,0.9);
        }}
        .modal-content {{
            display: block;
            margin: 20px auto;
            max-width: 95%;
            max-height: 90vh;
        }}
        .modal-close {{
            position: fixed;
            top: 20px; right: 40px;
            color: white;
            font-size: 50px;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <div id="imageModal" class="modal" onclick="this.style.display='none'">
        <span class="modal-close">&times;</span>
        <img class="modal-content" id="modalImg">
    </div>
    <script>
        function openModal(img) {{
            document.getElementById('imageModal').style.display = 'block';
            document.getElementById('modalImg').src = img.src;
        }}
    </script>

    <div class="container">
        <header>
            <h1>🔬 Geology Cross-Section Review Report</h1>
            <p>지질 단면도 검토 보고서</p>
            <p style="color: #666; margin-top: 5px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

            <div class="score-overview">
                <div class="score-card">
                    <div class="big-number">{avg_score:.1f}</div>
                    <div class="label">평균 점수 / 10</div>
                </div>
                <div class="score-card">
                    <div class="big-number">{len(self.reviews)}</div>
                    <div class="label">검토된 단면</div>
                </div>
                <div class="score-card">
                    <div class="big-number" style="color: var(--danger);">{len(all_critical)}</div>
                    <div class="label">중요 문제점</div>
                </div>
            </div>
        </header>

        <div class="section">
            <h2>⚠️ Critical Issues Summary (중요 문제점)</h2>
"""

        # Group issues by severity
        for severity in ['critical', 'major', 'minor']:
            issues = [i for i in all_critical if i.get('severity') == severity]
            if issues:
                for issue in issues:
                    issue_class = f"{severity}-issue"
                    html += f"""
            <div class="{issue_class}">
                <div class="issue-header">
                    <span class="severity-badge" style="background: {severity_color(severity)};">{severity.upper()}</span>
                    <span class="section-badge">{issue.get('section', '?')}</span>
                </div>
                <strong>{issue.get('issue', '')}</strong>
                <p style="margin-top: 8px; color: #666;">
                    <strong>위치:</strong> {issue.get('location', '?')}<br>
                    <strong>원리:</strong> {issue.get('geological_principle', '?')}
                </p>
            </div>
"""

        html += """
        </div>

        <div class="section">
            <h2>📊 Section-by-Section Reviews (단면별 상세 검토)</h2>
"""

        # Per-section reviews
        for r in self.reviews:
            section_name = r.get('section_name', 'Unknown')
            score = r.get('overall_score', 0)
            assessment = r.get('overall_assessment', '?')
            summary = r.get('summary_korean', '')
            image_path = r.get('image_path', '')

            html += f"""
            <div class="review-card">
                <div class="review-header">
                    <h3>{section_name}</h3>
                    <span class="review-score" style="color: {score_color(score)};">{score}/10 ({assessment})</span>
                </div>
                <div class="review-body">
"""
            # Image preview
            if image_path:
                import base64
                try:
                    with open(image_path, 'rb') as f:
                        img_data = base64.standard_b64encode(f.read()).decode('utf-8')
                    html += f"""
                    <div class="image-preview">
                        <img src="data:image/png;base64,{img_data}" alt="{section_name}" onclick="openModal(this)">
                    </div>
"""
                except:
                    pass

            html += f"""
                    <div class="review-summary">{summary}</div>
"""

            # Alluvium review
            alluvium = r.get('alluvium_review', {})
            if not alluvium.get('is_correct', True):
                html += """
                    <div class="sub-section">
                        <h4>🏔️ 충적층 문제 (Alluvium Issues)</h4>
"""
                for issue in alluvium.get('issues', []):
                    html += f'<p style="color: var(--danger);">• {issue}</p>'
                html += "</div>"

            # Recommendations
            recs = r.get('specific_recommendations', [])
            if recs:
                html += """
                    <div class="sub-section">
                        <h4>💡 개선 권고사항 (Recommendations)</h4>
"""
                for rec in sorted(recs, key=lambda x: x.get('priority', 99)):
                    priority = rec.get('priority', '?')
                    html += f"""
                        <div class="recommendation">
                            <span class="priority-badge">P{priority}</span>
                            {rec.get('recommendation', '')}
                        </div>
"""
                html += "</div>"

            html += """
                </div>
            </div>
"""

        html += f"""
        </div>

        <footer>
            <p>Generated by Geology Review Agent | Model: {self.model}</p>
            <p>This report was created by an AI reviewer for quality assurance purposes.</p>
        </footer>
    </div>
</body>
</html>
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"HTML report generated: {output_path}")
        return output_path


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  GEOLOGY CROSS-SECTION REVIEW AGENT")
    print("  지질 단면도 검토 에이전트")
    print("=" * 70)

    # Initialize agent
    try:
        agent = GeologyReviewAgent()
        print("\nReview Agent initialized successfully.")
    except Exception as e:
        print(f"\nError: {e}")
        return None

    # Review all sections
    results = agent.review_all_sections()

    if results:
        # Generate text report
        report = agent.generate_report()
        print("\n" + report)

        # Export results
        agent.export_results()

        # Save report as text file
        report_path = OUTPUT_DIR / "review_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nText report saved to: {report_path}")

        # Generate HTML report
        html_path = agent.generate_html_report()
        print(f"HTML report saved to: {html_path}")

    print("\n" + "=" * 70)
    print("  REVIEW COMPLETE!")
    print("=" * 70)

    return agent


if __name__ == "__main__":
    main()
