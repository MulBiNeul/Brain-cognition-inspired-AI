# Tower of Hanoi Benchmark

초기 배치·이동 제한·목표 기둥이 변하는 하노이탑으로 적응과 지속적 오류를 평가합니다.

- 모델 평가: 3유형 × 100문제, 총 300문제
- 사람 평가: 3유형, 총 10문제 (4/3/3)
- 기본 설정: 문제당 최대 5라운드, T=10, k=2 (R1 포함), 네 가지 피드백

## 설치

Python 3.11 이상이 필요합니다. 저장소 루트에서 실행합니다.

```bash
cd tower-of-hanoi
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 모델 실행

실행할 모델을 골라 해당 명령을 사용합니다. API 키는 현재 터미널에서 유지됩니다.

```bash
# GPT-6.1 Sol
source ./keys.sh openai
.venv/bin/python run.py --profile sol --output runs/sol_300
```

```bash
# Claude Opus 5.5
source ./keys.sh anthropic
.venv/bin/python run.py --profile opus --output runs/opus_300
```

```bash
# Claude Sonnet 5.5
source ./keys.sh anthropic
.venv/bin/python run.py --profile sonnet --output runs/sonnet_300
```

모델·T·k 등은 [configs/](configs/)에서 변경합니다. 결과는 지정한 `runs/` 폴더에 저장됩니다.

## 재개 및 로그 확인

재개할 결과 폴더를 지정합니다. 새 터미널에서는 해당 모델의 API 키를 먼저 등록합니다.

```bash
.venv/bin/python run.py --resume runs/sol_300
.venv/bin/python view_logs.py
```

## 사람 평가

[Hanoi_Human_Evaluation.html](Hanoi_Human_Evaluation.html)을 브라우저에서 엽니다. 설치나 API 키 없이 진행하며, 결과를 JSON으로 저장하고 불러올 수 있습니다.

## 상세 문서

[실험 설정](docs/experiment.md) · [데이터 생성·검증](docs/dataset.md) · [지표 정의](docs/metrics.md) · [사람 평가](docs/human.md)
