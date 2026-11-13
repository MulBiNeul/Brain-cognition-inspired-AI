# SameGame

문자열의 묶음을 지우며 피드백으로 숨겨진 목표와 규칙에 적응하는 추론 태스크다. 모델 평가와 사람 평가가 같은 문제 선택·채점·지속 오류 집계를 사용한다.

## 문제 구성

50문제에 각각 5라운드가 있다. 다른 문자 전환 금지 유형과 같은 문자 반복 삭제 금지 유형을 각각 25문제씩 사용한다.

| 항목 | 설정 |
|---|---|
| 입력 문자열 | A–E를 모두 사용하는 28–32글자, 일부 후보에 공백 포함 |
| 목표 | 공백을 제외하고 0·1·2·3글자 중 하나 남기기 |
| 다른 문자 규칙 | 예: `AC` — A 삭제 직후 C 삭제 금지 |
| 같은 문자 규칙 | 예: `BB` — B 삭제 직후 다시 B 삭제 금지 |
| 시도별 변화 | 문자열과 목표 길이가 모두 직전 실제 시도와 달라짐 |
| 라운드별 변화 | 규칙은 라운드 안에서 고정, 인접 라운드끼리는 서로 다름 |
| 공개 정보 | 시작 문자열, 조작 지시문, 제출 후 피드백 |
| 비공개 정보 | 목표 길이, 금지 규칙 |
| 종료 기준 | 최대 T=10회 안에 k=2회 연속 정답 |

문자열·목표·규칙의 비연속 재등장은 허용한다. 라운드에는 목표별 3개씩 총 12개 후보가 있으며, 전체 3,000개다. 다음 후보는 실제 풀이 제출 후 선택한다. 라운드나 문제 경계에서도 직전 입력·목표와 같아지지 않는다. 통신 재시도는 동일한 후보를 사용하며 풀이 시도를 소모하지 않는다.

T회 안에 성공하지 못하면 해당 문제의 남은 라운드를 건너뛰고 다음 문제로 이동한다. 한 문제 안에서는 응답·피드백과 API의 대화 상태를 이어가고, 다음 문제에서는 새 대화를 시작한다. 목표도 매 시도 바뀌면서 비공개이므로, 연속 성공에는 규칙 적응 외에 목표 선택의 영향도 포함된다.

## 삭제와 답안

붙어 있는 동일한 문자 2개 이상의 최대 묶음 전체를 지운다. `AAA`는 한 행동으로 모두 지운다. 남은 문자는 붙지만 공백을 넘어서는 합쳐지지 않는다. 새로 붙은 묶음은 다음 행동으로 지운다.

각 문자의 삭제 가능한 묶음에 왼쪽부터 번호를 붙인다. `A2`는 현재 두 번째 A 묶음이며, 삭제할 때마다 번호를 다시 센다. 금지 규칙을 위반했더라도 삭제는 계속한다. 지울 묶음이 없는 상태에서 실제 최종 문자열을 제출한다. `A A`는 2글자, `AB A`는 3글자다.

```json
{"moves": "B1 A1", "answer": ""}
```

예를 들어 `ABBBA`에서 `B1`을 지우면 `AA`, 이어서 `A1`을 지우면 빈 문자열이다. 내부 공백은 유지하고 양끝 공백은 제거한다. 모델에게 주는 영문 지시문과 조작 예시의 원본은 `src/evaluate.py`의 `INSTRUCTIONS`다.

피드백은 네 종류다. 규칙 위반 시 최초 위반 위치 하나만 덧붙이며, 금지쌍의 두 번째 행동을 위반 위치로 표시한다. 모든 위반 위치는 평가자용 결과에 보관한다.

- `Correct`
- `Correct, but rule violation. Rule violation: move 3.`
- `Incorrect, also rule violation. Rule violation: move 4.`
- `Incorrect, but no rule violation`

`Correct, but rule violation`은 목표 길이는 맞았지만 현재 규칙을 위반했다는 뜻이며, 연속 정답에는 포함되지 않는다.

## 지속 오류와 MPE

현재 정답은 기본 조작이 유효하고, 더 지울 묶음이 없으며, 제출한 최종 문자열이 실제 결과와 같고, 현재 목표와 현재 규칙을 모두 만족하는 답안이다.

```text
지속 오류 = 기본 조작이 유효한 완결된 답안
           AND 실제 최종 문자열을 정확히 제출
           AND 직전 라운드의 금지 규칙을 준수
           AND 현재 문제는 오답
```

판정에는 현재 시도의 문자열과 목표를 사용한다. 직전 문자열로 되돌리거나 직전 목표를 적용하지 않는다. 현재 목표를 못 맞춘 경우도 위 조건을 만족하면 지속 오류다. 현재 정답은 직전 규칙 위반 여부와 관계없이 지속 오류가 아니다. 형식 오류, 불가능한 삭제, 미완결 답안, 최종 문자열 허위 보고는 지속 오류로 세지 않는다.

- 라운드 PER = 지속 오류 횟수 / 최대 시도 횟수 T. 조기 성공해도 분모는 T다.
- 문제별 MPE = 실제 제출이 있는 R2–R5의 PER 평균.
- 전체 MPE = 집계 가능한 문제별 MPE의 평균.
- R1과 미진입 라운드는 제외한다. 집계 대상 문제·라운드 수를 함께 보고한다.
- 실제 제출 횟수를 분모로 한 진단값은 `observed_per`, `observed_mpe`로 구분한다.

**지속 오류 정의와 PER/MPE 계산은 사용자의 구체적인 사전 승인 없이 변경하지 않는다.** `tests/test_metric_contract.py`가 채점·집계의 보호 기준을 검사한다.

## 문제 검토와 경우의 수

`web/review.html`을 브라우저에서 열면 문제 → 라운드 → 후보를 선택해 직접 풀 수 있다. 비공개 목표·규칙과 정답·지속 오류·일반 오답 예시가 보이는 연구자용 화면이다.

| 지표 | 의미 |
|---|---|
| 도달 가능한 최종 조합 수 | 목표·규칙과 관계없이 더 지울 묶음이 없을 때까지 삭제한 묶음 조합의 수 |
| 목표 글자 수에 맞는 조합 | 위 조합 중 최종 글자 수가 목표와 일치하는 조합의 수. 규칙 위반을 포함 |
| 최종 답안 종류 | 도달 가능한 서로 다른 최종 문자열의 수. 목표·규칙과 무관 |

동일한 원래 문자들을 같은 묶음으로 지웠으면 삭제 순서만 다른 경우는 하나로 센다. 최종 문자열에는 내부 공백도 반영한다. 유형별 실험에서는 세 지표와 목표 조합 비율을 비교하며, MPE와 함께 성공률·시도 수·전환 라운드 진입 범위를 확인한다.

```bash
python scripts/build_review.py
python -m src.grade SG-001 1 --case c01 --group-counts
```

## 모델 평가

모든 명령은 `SameGame` 폴더에서 실행한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

macOS zsh에서 키를 화면에 표시하지 않고 입력한다. 사용하는 제공자의 키만 설정하면 된다.

```zsh
read -s 'OPENAI_API_KEY?OpenAI API key: '; echo
export OPENAI_API_KEY
read -s 'ANTHROPIC_API_KEY?Anthropic API key: '; echo
export ANTHROPIC_API_KEY
```

OpenAI 설정은 `configs/openai.toml`, Anthropic 설정은 `configs/anthropic.toml`이다. 프로필은 `sol`, `opus`, `sonnet`이다. 기본 reasoning effort는 medium, 최대 출력은 25,000토큰, 요청 timeout은 600초, 통신 재시도는 최대 10회다.

```bash
python run.py --profile sol --dry-run
python run.py --profile sol --reasoning-summary --output runs/sol
python run.py --profile opus --output runs/opus
python run.py --profile sonnet --output runs/sonnet
```

`--dry-run`은 API 호출 없이 첫 요청을 확인한다. 새 실행은 비어 있는 결과 폴더를 사용한다. 문제 선택에는 `--item SG-001` 또는 `--limit 5`, 다른 문제 파일에는 `--dataset PATH`를 지정한다.

```bash
python run.py --resume runs/sol
python run.py --resume runs/sol --timeout 900
python scripts/check_errors.py runs/sol
```

재개 시 저장된 문제·seed·응답을 재생해 대화 상태와 다음 후보를 복원한다. 완료된 응답을 재사용하며, 실행 코드가 달라지면 재개를 중단한다. `--reasoning-summary`는 OpenAI가 제공하는 진단용 요약을 저장한다. 요약의 부재만으로 대화 상태 전달 실패를 판단하지 않는다.

## 사람 평가

```bash
python human_server.py --output runs/human
```

`http://127.0.0.1:8765`에서 진행한다. 목표·규칙은 공개하지 않고, 같은 라운드에서 제출했던 각 시도의 시작 문자열과 삭제 단계를 수직 기록으로 보여준다. 서버에 제출·피드백·집계를 저장하며, 통신 재시도는 중복 제출로 세지 않는다.

## 생성과 검증

```bash
python scripts/generate.py --count 50 --workers 4
python scripts/generate.py --validate-only
python scripts/build_review.py
python -m unittest discover -s tests -v
```

생성기는 현재 정답과 다른 목표에 도달하는 수순을 검사한다. 모든 전환 후보에는 지속 오류·일반 오답 수순이 있다. 목표에 도달하는 모든 물리적 수순에는 병합 후 삭제가 최소 2회 필요하다. 1–3글자 목표에는 서로 다른 문자 구성의 정답이 최소 2종류 있다. 같은 문자 유형은 삭제 문자를 계속 번갈아 고르는 수순만으로 목표에 도달할 수 없다.

같은 라운드 안에서는 삭제 구조를 중복하지 않으며, 전체에서 한 구조의 사용은 최대 2회다. `data/seed_structures.json`은 같은 문자 유형 생성에 필요한 구조 재료다. 생성·검증 결과는 `data/design_report.json`에 기록한다. 생성 명령은 지정한 출력 디렉터리의 데이터 파일을 갱신한다.

300문제로 확장할 때는 별도 출력 경로를 지정할 수 있다. 유형별 150문제, 총 18,000개 후보를 생성한다.

```bash
python scripts/generate.py --count 300 --workers 4 --output-dir data/catalog_300
python run.py --dataset data/catalog_300/problems.json --profile sol --output runs/sol_300
```

## 폴더 구조

```text
SameGame/
  run.py                 모델 평가 실행·재개
  human_server.py        사람 평가 서버
  configs/               제공자별 모델·실험 설정
  data/                  문제, 정답, 검증 보고서, 생성용 구조
  src/
    grade.py             문자열 조작·탐색·후보별 검사
    evaluate.py          채점·피드백·PER/MPE·진행 기준
    session.py           시도별 후보 선택·재생
    catalog.py           문제 파일 읽기·형식 검사
    dataset.py           문제 생성
    dataset_checks.py    구조·풀이·도달성 검증
    model_runner.py      API 실행·재시도·저장·터미널 로그
    human_server.py      사람 평가 세션·HTTP
    providers.py         제공자 API 연결
    diagnostics.py       추론 요약
    error_log.py         오류 기록·조회
  scripts/               생성·검토판 빌드·오류 조회 명령
  web/                   연구자 검토판·사람 평가 화면
  tests/                 기능·지표 보호 테스트
  runs/                  실행 결과, Git 제외
```
