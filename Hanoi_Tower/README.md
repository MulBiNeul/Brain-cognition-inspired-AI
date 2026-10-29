# Tower of Hanoi Cognitive Flexibility Benchmark

초기 배치, 이동 제한, 목표 기둥이 변하는 하노이탑에서 적응과 지속적 오류를 평가합니다. 참가자는 전체 이동 계획을 제출하고 네 가지 피드백을 받습니다. 목표 기둥·추가 제한의 내용·라운드 전환·T·k는 참가자에게 공개하지 않습니다.

| 평가 | 구성 | 기본 설정 |
|---|---|---|
| 모델 | 세 유형 각 100개, 총 300문제를 모델마다 실행 | Sol / Opus / Sonnet, medium, T=10, k=2 |
| 사람 | 세 유형 4/3/3, 총 10문제 | T=10, k=2, HTML에서 기둥 클릭 |

문제당 최대 5라운드이며 R1에도 k를 적용합니다. 모델은 **P001–P100 기둥 이동 제한 → P101–P200 연속 출발 제한 → P201–P300 선행 방문 조건** 순서로 진행합니다. 사람용 표본은 모델용 300문제의 동일 ID·내용을 사용합니다.

## 설치

모델 평가와 데이터·HTML 작업에는 Python **3.11 이상**이 필요합니다. `Hanoi_Tower` 폴더에서 실행합니다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

사람 평가는 포함된 HTML 파일을 브라우저에서 열면 되며 설치나 API 키가 필요하지 않습니다.

## 모델 평가

각 모델을 별도 터미널에서 실행합니다. API 키는 현재 터미널의 환경 변수에만 등록하며, 실행 전에 인증과 모델 접근을 확인합니다.

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

세 모델은 같은 `data/benchmark_300.json`을 사용합니다. 모델·effort·토큰 한도·T·k는 `configs/sol.toml`, `configs/opus.toml`, `configs/sonnet.toml`에서 변경합니다. 기본 출력 한도는 25,000토큰입니다.

```bash
# Check authentication and model access without generation
.venv/bin/python run.py --profile sol --check-only

# Preview requests without API calls or keys
.venv/bin/python run.py --profile sol --dry-run

# Resume the saved run
.venv/bin/python run.py --resume runs/sol_300
.venv/bin/python run.py --resume runs/opus_300
.venv/bin/python run.py --resume runs/sonnet_300
```

재개는 저장된 문제·설정·코드의 일치를 검사한 뒤 대화·추론 상태·연속 정답 횟수를 복원합니다. 완료된 응답은 재사용하고, 응답이 저장되지 않은 요청만 다시 보냅니다. 해당 요청은 서버에서 이미 처리되었을 가능성이 있어 비용이 중복될 수 있습니다. 재개 후 통신 오류가 다시 발생하면 멈추며 자동 반복 요청은 하지 않습니다.

실행 중인 실험의 코드·문제·설정은 바꾸지 않습니다. 기존 결과 폴더는 새 실행으로 덮어쓰지 않으며, 새 실험에는 별도 `--output` 경로를 지정합니다.

## 사람 평가

**[Hanoi_Human_Evaluation.html](Hanoi_Human_Evaluation.html)을 브라우저에서 엽니다.** 이 파일 하나만 참가자에게 전달하면 됩니다. 서버 실행과 인터넷 연결은 필요하지 않습니다.

참여 코드를 입력하고 기둥을 클릭해 계획을 작성합니다. 같은 문제의 이전 계획과 피드백을 다시 볼 수 있습니다. 중간에는 **진행 저장 · JSON**, 완료 후에는 **결과 저장 · JSON**으로 내려받고 연구자에게 전달합니다. 같은 HTML에서 **불러오기**로 JSON을 선택하면 다른 브라우저에서도 이어갈 수 있습니다.

`configs/human.toml`이나 `data/human_10.json`을 변경했다면 HTML을 다시 생성합니다.

```bash
python3 build_human.py
```

사람과 모델 모두 **1~30회 이동**을 제출합니다. 사람의 클릭 입력은 기본 이동 오류를 막으므로 모델의 자유 JSON 입력과 입력 방식이 다릅니다. 자세한 안내는 [사람 평가](docs/human.md)에 있습니다.

## 채점과 결과

`정답 / 오답 / 규칙 위반 / 정답이지만 규칙 위반`의 네 피드백을 사용합니다. 정답만 연속 정답 횟수를 늘리고, 나머지는 0으로 초기화합니다. T 안에 k를 충족하지 못하면 그 문제의 남은 라운드를 제외하고 다음 문제로 넘어갑니다.

- **Round PER**: R2–R5에서 지속적 오류 수 / 실제 채점 제출 수.
- **Problem MPE**: 해당 문제에서 진행한 R2–R5 PER의 평균.
- **Total MPE**: 계산 가능한 문제들의 MPE 평균.
- **평균 오답률**: R1–R5 라운드 오답률 → 문제 평균 → 전체 문제 평균.

모델 결과 폴더에는 설정, 제출·채점 기록, 집계, 원본 API 요청·응답, 제공된 추론 요약을 저장합니다. 추론 요약은 내부 추론 전체를 의미하지 않습니다. 사람 결과는 HTML에서 JSON 파일로 내려받습니다.

```bash
# Open the researcher log viewer
.venv/bin/python view_logs.py
```

세부 기준은 [실험 설정](docs/experiment.md)과 [지표 정의](docs/metrics.md)를 참고하세요.

## 데이터 생성과 검증

현재 실험에는 제공된 `data/benchmark_300.json`과 `data/human_10.json`을 그대로 사용합니다. 새 후보가 필요한 경우에만 생성기를 실행합니다. 생성기는 규칙 템플릿과 로컬 탐색을 사용하며 LLM API를 호출하지 않습니다.

```bash
# Validate the active 300-problem dataset
python3 validate_problems.py
python3 validate_problems.py data/human_10.json

# Generate separate candidates; this does not reproduce the fixed release bank
python3 generate_problems.py --types peg source_sequence visit --per-type 100 --seed 20261029 --output generated/problems.json
```

생성된 파일은 현재 실험 자료를 자동으로 바꾸지 않습니다. 데이터 형식과 유형 설명은 [데이터셋](docs/dataset.md)에 있습니다.

API 키(`.env*`), 가상환경, 실험 결과(`runs/`), 별도로 생성한 후보(`generated/`)는 Git에서 제외합니다.
