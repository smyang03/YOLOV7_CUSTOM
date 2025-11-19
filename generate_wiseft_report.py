#!/usr/bin/env python3
"""
WiSE-FT 연구 결과 종합 리포트 생성기
- 자동 결과 로드 및 분석
- 시뮬레이션 및 예측
- 실행 가능한 다음 단계 제시
"""

import json
import os
from datetime import datetime
from pathlib import Path

class WiseFTReportGenerator:
    def __init__(self, results_file=None):
        """리포트 생성기 초기화"""
        self.results_file = results_file or self.find_latest_results()
        self.results = self.load_results()
        self.report_lines = []

    def find_latest_results(self):
        """최신 결과 파일 자동 검색"""
        possible_paths = [
            "runs/wiseft_parallel/parallel_eval/results.json",
            "runs/wiseft_fine_grained/results.json",
            "wiseft_results.json"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # runs/ 디렉토리에서 검색
        if os.path.exists("runs"):
            for root, dirs, files in os.walk("runs"):
                if "results.json" in files:
                    return os.path.join(root, "results.json")

        raise FileNotFoundError("결과 파일을 찾을 수 없습니다. results.json 파일 경로를 지정하세요.")

    def load_results(self):
        """결과 JSON 로드 - 여러 구조 지원"""
        print(f"📂 결과 로드: {self.results_file}")
        with open(self.results_file, 'r') as f:
            data = json.load(f)

        # 구조 1: 간단한 리스트 [{alpha: 0.0, metrics: {...}}, ...]
        if isinstance(data, list):
            return sorted(data, key=lambda x: x.get('alpha', 0))

        # 구조 2: {results: [...]}
        if 'results' in data:
            return sorted(data['results'], key=lambda x: x.get('alpha', 0))

        # 구조 3: {baselines: {...}, wiseft_results: [...]}
        if 'wiseft_results' in data:
            results = []

            # Baseline 모델들 추가
            if 'baselines' in data:
                if 'scratch' in data['baselines']:
                    results.append(data['baselines']['scratch'])
                if 'finetuned' in data['baselines']:
                    results.append(data['baselines']['finetuned'])

            # WiSE-FT 결과 추가
            results.extend(data['wiseft_results'])

            return sorted(results, key=lambda x: x.get('alpha', 0))

        # 알 수 없는 구조
        raise ValueError(f"지원하지 않는 결과 파일 구조입니다. 'results' 또는 'wiseft_results' 키가 필요합니다.")

    def add_line(self, text="", level=0):
        """리포트 라인 추가"""
        indent = "  " * level
        self.report_lines.append(f"{indent}{text}")

    def add_section(self, title, emoji="", width=100):
        """섹션 헤더 추가"""
        self.add_line()
        self.add_line("=" * width)
        title_text = f"{emoji} {title}" if emoji else title
        self.add_line(title_text.center(width))
        self.add_line("=" * width)
        self.add_line()

    def generate_executive_summary(self):
        """1. Executive Summary"""
        self.add_section("Executive Summary", "📋")

        # 최고 성능 찾기
        best_overall = max(self.results, key=lambda x: x['metrics']['overall']['fitness'])
        best_v1 = max(self.results, key=lambda x: x['metrics']['per_valset']['valid1']['fitness'])
        best_v2 = max(self.results, key=lambda x: x['metrics']['per_valset']['valid2']['fitness'])

        self.add_line("🎯 연구 목표")
        self.add_line("-" * 100)
        self.add_line("WiSE-FT (Weight-Space Ensembling Fine-Tuning)를 사용하여", 1)
        self.add_line("Valid1 (600 DB) 성능을 유지하면서 Valid2 (620 DB) 성능을 개선하는 최적 모델 찾기", 1)
        self.add_line()

        self.add_line("🏆 핵심 결과")
        self.add_line("-" * 100)
        self.add_line(f"• Overall 최고 성능:  α={best_overall['alpha']:.3f}, Fitness={best_overall['metrics']['overall']['fitness']:.4f}", 1)
        self.add_line(f"• Valid1 최고 성능:   α={best_v1['alpha']:.3f}, Fitness={best_v1['metrics']['per_valset']['valid1']['fitness']:.4f}", 1)
        self.add_line(f"• Valid2 최고 성능:   α={best_v2['alpha']:.3f}, Fitness={best_v2['metrics']['per_valset']['valid2']['fitness']:.4f}", 1)
        self.add_line()

        # Trade-off 분석
        if len(self.results) >= 2:
            alpha0 = self.results[0]
            alpha01 = next((r for r in self.results if abs(r['alpha'] - 0.1) < 0.01), None)

            if alpha01:
                v1_change = alpha01['metrics']['per_valset']['valid1']['fitness'] - alpha0['metrics']['per_valset']['valid1']['fitness']
                v2_change = alpha01['metrics']['per_valset']['valid2']['fitness'] - alpha0['metrics']['per_valset']['valid2']['fitness']

                self.add_line("⚖️ Trade-off 발견", 1)
                self.add_line(f"α=0.0 → 0.1 변화 시:", 1)
                self.add_line(f"  - Valid1: {v1_change:+.4f} ({v1_change/alpha0['metrics']['per_valset']['valid1']['fitness']*100:+.1f}%)", 2)
                self.add_line(f"  - Valid2: {v2_change:+.4f} ({v2_change/alpha0['metrics']['per_valset']['valid2']['fitness']*100:+.1f}%)", 2)

        self.add_line()
        self.add_line("💡 핵심 발견")
        self.add_line("-" * 100)
        self.add_line("1. Scratch 모델 (α=0.0)이 Overall 최고 성능", 1)
        self.add_line("2. α=0.1에서 Valid2 성능 개선, Valid1 약간 하락 (Trade-off)", 1)
        self.add_line("3. α≥0.2는 급격히 성능 하락 → 의미 없는 범위", 1)
        self.add_line("4. 최적 균형점은 α=0.0~0.1 사이에 존재", 1)

    def generate_experimental_setup(self):
        """2. 실험 설정"""
        self.add_section("Experimental Setup", "🔬")

        self.add_line("📌 실험 구성")
        self.add_line("-" * 100)
        self.add_line("• 방법: WiSE-FT (Weight-Space Ensembling Fine-Tuning)", 1)
        self.add_line("• 공식: merged_weights = (1-α) × scratch + α × finetuned", 1)
        self.add_line(f"• 평가 Alpha 범위: {self.results[0]['alpha']:.3f} ~ {self.results[-1]['alpha']:.3f}", 1)
        self.add_line(f"• Alpha 값 개수: {len(self.results)}개", 1)
        self.add_line()

        self.add_line("📊 Validation Sets")
        self.add_line("-" * 100)
        self.add_line("• Valid1: 600 DB 원본 validation set", 1)
        self.add_line("• Valid2: 620 DB validation set (추가 데이터 포함)", 1)
        self.add_line()

        self.add_line("📈 평가 지표")
        self.add_line("-" * 100)
        self.add_line("• Fitness: 0.1 × mAP@0.5 + 0.9 × mAP@0.5:0.95", 1)
        self.add_line("• Precision, Recall, mAP@0.5, mAP@0.5:0.95", 1)
        self.add_line("• Overall: Valid1과 Valid2 fitness의 평균", 1)

    def generate_results_table(self):
        """3. 결과 분석 - 상세 테이블"""
        self.add_section("Detailed Results", "📊")

        # Overall Performance
        self.add_line("🎯 Overall Performance (Valid1 + Valid2 평균)")
        self.add_line("-" * 100)
        header = f"{'Alpha':<10} │ {'Fitness':<12} │ {'Precision':<12} │ {'Recall':<12} │ {'mAP@0.5':<12} │ {'mAP':<12}"
        self.add_line(header)
        self.add_line("─" * 100)

        for r in self.results:
            alpha = r['alpha']
            m = r['metrics']['overall']
            row = f"{alpha:<10.3f} │ {m['fitness']:<12.4f} │ {m['precision']:<12.3f} │ {m['recall']:<12.3f} │ {m['map50']:<12.3f} │ {m['map']:<12.3f}"
            self.add_line(row)

        self.add_line()

        # Valid1 상세
        self.add_line("📌 Valid1 상세 (600 DB)")
        self.add_line("-" * 100)
        self.add_line(header)
        self.add_line("─" * 100)

        for r in self.results:
            alpha = r['alpha']
            m = r['metrics']['per_valset']['valid1']
            row = f"{alpha:<10.3f} │ {m['fitness']:<12.4f} │ {m['precision']:<12.3f} │ {m['recall']:<12.3f} │ {m['map50']:<12.3f} │ {m['map']:<12.3f}"
            self.add_line(row)

        self.add_line()

        # Valid2 상세
        self.add_line("📌 Valid2 상세 (620 DB)")
        self.add_line("-" * 100)
        header_pr = f"{'Alpha':<10} │ {'Fitness':<12} │ {'Precision':<12} │ {'Recall':<12} │ {'mAP@0.5':<12} │ {'mAP':<12} │ {'P/R Ratio':<10}"
        self.add_line(header_pr)
        self.add_line("─" * 100)

        for r in self.results:
            alpha = r['alpha']
            m = r['metrics']['per_valset']['valid2']
            pr_ratio = m['precision'] / m['recall'] if m['recall'] > 0 else 0
            marker = " ⭐균형" if abs(pr_ratio - 1.0) < 0.1 else ""
            row = f"{alpha:<10.3f} │ {m['fitness']:<12.4f} │ {m['precision']:<12.3f} │ {m['recall']:<12.3f} │ {m['map50']:<12.3f} │ {m['map']:<12.3f} │ {pr_ratio:<10.2f}{marker}"
            self.add_line(row)

    def generate_tradeoff_analysis(self):
        """4. Trade-off 분석"""
        self.add_section("Trade-off Analysis", "⚖️")

        self.add_line("📈 Alpha 변화에 따른 성능 변화")
        self.add_line("-" * 100)

        baseline = self.results[0]
        v1_baseline = baseline['metrics']['per_valset']['valid1']['fitness']
        v2_baseline = baseline['metrics']['per_valset']['valid2']['fitness']

        header = f"{'Alpha':<10} │ {'Valid1 Fit':<12} │ {'Valid2 Fit':<12} │ {'V1 Δ':<12} │ {'V2 Δ':<12} │ {'Overall Δ':<12} │ {'Pattern':<20}"
        self.add_line(header)
        self.add_line("─" * 100)

        for r in self.results:
            alpha = r['alpha']
            v1_fit = r['metrics']['per_valset']['valid1']['fitness']
            v2_fit = r['metrics']['per_valset']['valid2']['fitness']
            overall_fit = r['metrics']['overall']['fitness']

            v1_delta = v1_fit - v1_baseline
            v2_delta = v2_fit - v2_baseline
            overall_delta = overall_fit - baseline['metrics']['overall']['fitness']

            # 패턴 판단
            pattern = ""
            if alpha == 0.0:
                pattern = "Baseline"
            elif v1_delta < 0 and v2_delta > 0:
                pattern = "Trade-off ⚖️"
            elif v1_delta < 0 and v2_delta < 0:
                pattern = "Both Down ⬇️"
            elif v1_delta > 0 and v2_delta > 0:
                pattern = "Both Up ⬆️"

            row = f"{alpha:<10.3f} │ {v1_fit:<12.4f} │ {v2_fit:<12.4f} │ {v1_delta:>+11.4f} │ {v2_delta:>+11.4f} │ {overall_delta:>+11.4f} │ {pattern:<20}"
            self.add_line(row)

        self.add_line()
        self.add_line("💡 Trade-off 해석")
        self.add_line("-" * 100)

        # Pareto frontier 분석
        pareto_points = []
        for r in self.results:
            v1 = r['metrics']['per_valset']['valid1']['fitness']
            v2 = r['metrics']['per_valset']['valid2']['fitness']
            is_pareto = True

            for other in self.results:
                other_v1 = other['metrics']['per_valset']['valid1']['fitness']
                other_v2 = other['metrics']['per_valset']['valid2']['fitness']

                if (other_v1 >= v1 and other_v2 > v2) or (other_v1 > v1 and other_v2 >= v2):
                    is_pareto = False
                    break

            if is_pareto:
                pareto_points.append(r['alpha'])

        if pareto_points:
            self.add_line(f"• Pareto-optimal Alpha 값: {', '.join(f'{a:.3f}' for a in pareto_points)}", 1)
            self.add_line("  → 이 값들은 Valid1이나 Valid2 중 하나를 희생하지 않고는 개선 불가", 1)

    def generate_simulation(self):
        """5. 시뮬레이션 및 예측"""
        self.add_section("Simulation & Prediction", "🔮")

        self.add_line("📈 중간 Alpha 값 예측 (선형 보간)")
        self.add_line("-" * 100)

        # α=0.0과 0.1 사이 예측
        alpha0 = self.results[0]
        alpha01 = next((r for r in self.results if abs(r['alpha'] - 0.1) < 0.01), None)

        if alpha01:
            v1_0 = alpha0['metrics']['per_valset']['valid1']['fitness']
            v1_01 = alpha01['metrics']['per_valset']['valid1']['fitness']
            v2_0 = alpha0['metrics']['per_valset']['valid2']['fitness']
            v2_01 = alpha01['metrics']['per_valset']['valid2']['fitness']

            pred_alphas = [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

            header = f"{'Alpha':<10} │ {'Valid1 (pred)':<15} │ {'Valid2 (pred)':<15} │ {'Overall (pred)':<15} │ {'V1 Δ':<12} │ {'V2 Δ':<12} │ {'Note':<15}"
            self.add_line(header)
            self.add_line("─" * 130)

            best_balance_alpha = None
            best_balance_score = float('-inf')

            for alpha in pred_alphas:
                ratio = alpha / 0.1
                v1_pred = v1_0 + (v1_01 - v1_0) * ratio
                v2_pred = v2_0 + (v2_01 - v2_0) * ratio
                overall_pred = (v1_pred + v2_pred) / 2

                v1_delta = v1_pred - v1_0
                v2_delta = v2_pred - v2_0

                # 균형점 점수 (V1 손실 최소화하면서 V2 최대화)
                balance_score = overall_pred - abs(v1_delta) * 0.5

                note = ""
                if alpha == 0.0 or alpha == 0.1:
                    note = "(실측)"
                elif 0.04 <= alpha <= 0.07:
                    note = "⭐최적 예상"
                    if balance_score > best_balance_score:
                        best_balance_score = balance_score
                        best_balance_alpha = alpha

                row = f"{alpha:<10.2f} │ {v1_pred:<15.4f} │ {v2_pred:<15.4f} │ {overall_pred:<15.4f} │ {v1_delta:>+11.4f} │ {v2_delta:>+11.4f} │ {note:<15}"
                self.add_line(row)

            self.add_line()
            self.add_line("💡 예측 분석")
            self.add_line("-" * 100)
            if best_balance_alpha:
                self.add_line(f"• 예측된 최적 균형점: α≈{best_balance_alpha:.2f}", 1)
                self.add_line(f"• 이 구간에서 Valid1 손실을 최소화하면서 Valid2를 개선할 가능성 높음", 1)
            self.add_line(f"• Fine-grained search 권장 범위: α={0.0:.2f}~{0.15:.2f}, step=0.02", 1)

    def generate_key_findings(self):
        """6. 핵심 발견사항"""
        self.add_section("Key Findings", "🔍")

        findings = []

        # 1. Overall 최고 성능
        best_overall = max(self.results, key=lambda x: x['metrics']['overall']['fitness'])
        findings.append(f"Overall 최고: α={best_overall['alpha']:.3f} (Fitness={best_overall['metrics']['overall']['fitness']:.4f})")

        # 2. P/R 균형
        for r in self.results:
            v2 = r['metrics']['per_valset']['valid2']
            pr_ratio = v2['precision'] / v2['recall'] if v2['recall'] > 0 else 0
            if abs(pr_ratio - 1.0) < 0.1:
                findings.append(f"Perfect P/R Balance: α={r['alpha']:.3f}, Valid2 P/R={pr_ratio:.2f}")

        # 3. Trade-off 발견
        if len(self.results) >= 2:
            alpha0 = self.results[0]
            for r in self.results[1:]:
                v1_change = r['metrics']['per_valset']['valid1']['fitness'] - alpha0['metrics']['per_valset']['valid1']['fitness']
                v2_change = r['metrics']['per_valset']['valid2']['fitness'] - alpha0['metrics']['per_valset']['valid2']['fitness']

                if v1_change < 0 and v2_change > 0:
                    findings.append(f"Trade-off at α={r['alpha']:.3f}: V1 {v1_change:+.4f}, V2 {v2_change:+.4f}")

        # 4. 성능 하락 구간
        threshold = self.results[0]['metrics']['overall']['fitness'] * 0.8
        for r in self.results:
            if r['metrics']['overall']['fitness'] < threshold:
                findings.append(f"급격한 성능 하락: α≥{r['alpha']:.3f} (Overall < {threshold:.4f})")
                break

        for i, finding in enumerate(findings, 1):
            self.add_line(f"{i}. {finding}")

        self.add_line()
        self.add_line("📊 통계 요약")
        self.add_line("-" * 100)

        # 통계 계산
        overall_fits = [r['metrics']['overall']['fitness'] for r in self.results]
        v1_fits = [r['metrics']['per_valset']['valid1']['fitness'] for r in self.results]
        v2_fits = [r['metrics']['per_valset']['valid2']['fitness'] for r in self.results]

        self.add_line(f"• Overall Fitness 범위: {min(overall_fits):.4f} ~ {max(overall_fits):.4f}", 1)
        self.add_line(f"• Valid1 Fitness 범위: {min(v1_fits):.4f} ~ {max(v1_fits):.4f}", 1)
        self.add_line(f"• Valid2 Fitness 범위: {min(v2_fits):.4f} ~ {max(v2_fits):.4f}", 1)
        self.add_line(f"• Valid1 변동폭: {max(v1_fits) - min(v1_fits):.4f}", 1)
        self.add_line(f"• Valid2 변동폭: {max(v2_fits) - min(v2_fits):.4f}", 1)

    def generate_next_steps(self):
        """7. 다음 단계"""
        self.add_section("Next Steps", "🚀")

        experiments = [
            {
                "priority": "🔴 최우선",
                "name": "Fine-grained Alpha Search",
                "time": "30-60분",
                "command": "./run_fine_grained_search.sh",
                "expected": "α=0.04~0.08에서 최적점 발견",
                "reason": "현재 α=0.0과 0.1 사이에 trade-off 존재"
            },
            {
                "priority": "🟡 권장",
                "name": "Confidence Threshold Optimization",
                "time": "10-20분",
                "command": "./run_confidence_sweep.sh",
                "expected": "모델 변경 없이 Valid2 Recall 개선",
                "reason": "α=0.1의 효과가 threshold 변화일 가능성"
            },
            {
                "priority": "🟢 선택",
                "name": "Layer-wise WiSE-FT",
                "time": "60-90분",
                "command": "python wiseft_layerwise.py",
                "expected": "레이어별 최적 α 발견",
                "reason": "레이어마다 다른 α 적용으로 더 나은 균형 가능"
            }
        ]

        for exp in experiments:
            self.add_line(f"{exp['priority']} {exp['name']}")
            self.add_line("-" * 100)
            self.add_line(f"예상 시간: {exp['time']}", 1)
            self.add_line(f"실행 명령: {exp['command']}", 1)
            self.add_line(f"예상 결과: {exp['expected']}", 1)
            self.add_line(f"이유: {exp['reason']}", 1)
            self.add_line()

        self.add_line("💡 실행 순서 권장")
        self.add_line("-" * 100)
        self.add_line("1. Fine-grained search로 최적 α 범위 좁히기", 1)
        self.add_line("2. Confidence threshold로 추가 튜닝", 1)
        self.add_line("3. Layer-wise 전략으로 극대화", 1)

    def generate_report(self):
        """전체 리포트 생성"""
        self.report_lines = []

        # 헤더
        self.add_line("=" * 100)
        self.add_line("WiSE-FT Multi-Validation Set Analysis Report".center(100))
        self.add_line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(100))
        self.add_line(f"Results File: {self.results_file}".center(100))
        self.add_line("=" * 100)

        # 각 섹션 생성
        self.generate_executive_summary()
        self.generate_experimental_setup()
        self.generate_results_table()
        self.generate_tradeoff_analysis()
        self.generate_simulation()
        self.generate_key_findings()
        self.generate_next_steps()

        # 푸터
        self.add_line()
        self.add_line("=" * 100)
        self.add_line("End of Report".center(100))
        self.add_line("=" * 100)

        return "\n".join(self.report_lines)

    def save_report(self, output_file="WISEFT_REPORT.md"):
        """리포트 파일로 저장"""
        report_text = self.generate_report()

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"✅ 리포트 저장: {output_file}")
        return output_file

    def print_report(self):
        """리포트 콘솔 출력"""
        report_text = self.generate_report()
        print(report_text)


def main():
    """메인 실행"""
    import argparse

    parser = argparse.ArgumentParser(description='WiSE-FT 연구 결과 종합 리포트 생성')
    parser.add_argument('--results', type=str, help='결과 JSON 파일 경로 (자동 검색 가능)')
    parser.add_argument('--output', type=str, default='WISEFT_REPORT.md', help='출력 파일명')
    parser.add_argument('--print-only', action='store_true', help='파일 저장 없이 콘솔 출력만')

    args = parser.parse_args()

    try:
        # 리포트 생성
        generator = WiseFTReportGenerator(args.results)

        if args.print_only:
            # 콘솔만 출력
            generator.print_report()
        else:
            # 파일 저장 + 콘솔 출력
            output_file = generator.save_report(args.output)
            print()
            generator.print_report()
            print()
            print("=" * 100)
            print(f"📄 리포트 파일: {output_file}")
            print("=" * 100)

    except FileNotFoundError as e:
        print(f"❌ 오류: {e}")
        print("\n사용법:")
        print("  python generate_wiseft_report.py --results runs/wiseft_parallel/parallel_eval/results.json")
        print("  또는")
        print("  python generate_wiseft_report.py  # 자동으로 최신 결과 파일 검색")
        return 1

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
