# Study-AIText_Detection

짧은 텍스트에서 AI가 쓴 글인지 탐지하는 연구를 위한 스터디 레포. 두 가지 설정을 본다.

- **단일 문장**: 문장 하나의 임베딩이나 확률 통계만으로 판별
- **설문 응답**: 한 응답자가 여러 문항에 답한 묶음이 있을 때, 문항 간 일관성과 응답 분포까지 써서 판별

두 설정은 "텍스트가 짧아서 신호가 약하다"는 문제를 공유한다. 기존 detector들은 주로 긴 텍스트에서 성능을 보고하고, 짧은 텍스트에서는 약해진다고 알려져 있다. 이걸 먼저 직접 확인한다.

## 진행 상황

| 실험 | 상태 | 요약 |
|---|---|---|
| [01 Fast-DetectGPT 재현](experiments/01_fastdetectgpt_1b/README.md) | 진행 중 | GPT-Neo-2.7B로는 논문 수치가 재현된다. 학습 중 LLM 산출물에 노출된 최신 채점 모델(Llama-3.2, Gemma-3, SmolLM2)에서는 GPT-3.5/4 글의 entropy가 likelihood 상승을 상쇄해서, Fast-DetectGPT가 단순 Likelihood보다 낮아진다 |

논의와 결정 기록: [docs/research-log.md](docs/research-log.md)

## 진행 원칙

주제나 method를 먼저 정하지 않는다. 순서는 아래와 같다.

1. **재현**: 대표 선행연구 1~2편을 공개 코드와 데이터로 돌려본다.
2. **관찰**: 숫자를 맞추는 것보다, detector가 틀린 샘플을 100개 정도 직접 읽는 데 시간을 쓴다.
3. **실패 패턴**: 왜 틀렸는지가 유형으로 묶이는지 본다.
4. **Method**: 패턴이 보일 때만 그걸 겨냥한 방법을 고민한다. 패턴이 안 보이거나 재현이 안 되면 접는다.

## 재현하면서 확인할 것

- 텍스트를 줄여가면(문단 → 몇 문장 → 한 문장) zero-shot detector 성능이 어디서부터 무너지는가
- 생성 모델, 프롬프트, 디코딩 설정을 바꾸면 결과가 얼마나 흔들리는가
- 한 문장 설정에서 틀리는 샘플은 어떤 특성을 갖는가 (길이, 주제, 문체, 자주 쓰이는 표현)
- 설문 설정에서 한 사람의 응답 여러 개를 묶으면 응답 하나만 볼 때보다 얼마나 좋아지는가

## 선행연구 후보

### 탐지 방법 (1순위: 먼저 재현)

| 논문 | 학회 | 코드 | 고른 이유 |
|---|---|---|---|
| [Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature](https://arxiv.org/abs/2310.05130) | ICLR 2024 | [baoguangsheng/fast-detect-gpt](https://github.com/baoguangsheng/fast-detect-gpt) | DetectGPT의 섭동 단계를 샘플링으로 바꿔 빨라진 zero-shot 방법. 학습 없이 바로 돌려볼 수 있음 |
| [Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text](https://arxiv.org/abs/2401.12070) | ICML 2024 | [ahans30/Binoculars](https://github.com/ahans30/Binoculars) | 두 언어모델의 perplexity를 비교하는 zero-shot 방법. Fast-DetectGPT와 나란히 비교하기 좋음. 기본 설정은 7B 모델 2개라 작은 모델 쌍으로 대체할 수 있는지 확인 필요 |
| [DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature](https://arxiv.org/abs/2301.11305) | ICML 2023 | [eric-mitchell/detect-gpt](https://github.com/eric-mitchell/detect-gpt) | 확률 곡률 기반 탐지의 출발점. 느려서 참고용 |
| [When Less is More: Understanding When Token Filtering Helps and Fails in AI-generated Text Detection](https://aclanthology.org/2026.emnlp-main.500/) | EMNLP 2026 | 없음 | 정규화를 뺀 Fast-DetectGPT 점수(EGS)를 entropy calibration 관점에서 분석. 실험 01과 가장 가까운 선행연구 |

### 벤치마크

| 논문 | 학회 | 코드/데이터 | 고른 이유 |
|---|---|---|---|
| [RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors](https://aclanthology.org/2024.acl-long.674) | ACL 2024 | [liamdugan/raid](https://github.com/liamdugan/raid) | 생성 모델 11개, 도메인 8개, 공격 11종, 600만 개 이상의 생성문. 길이를 잘라 짧은 텍스트 실험을 만들기 좋음 |
| [DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios](https://arxiv.org/abs/2410.23746) | NeurIPS 2024 D&B | [junchaoIU/DetectRL](https://github.com/junchaoIU/DetectRL) | 실제 사용 환경에 가까운 설정의 벤치마크 |

### 설문 응답 설정

| 논문 | 학회 | 코드/데이터 | 고른 이유 |
|---|---|---|---|
| [Towards Detecting AI-Assisted Responses in Online Surveys](https://arxiv.org/abs/2609.17317) (ASURRE) | EMNLP 2026 | [mike-qz-wang/ASURRE](https://github.com/mike-qz-wang/ASURRE) | 실제 설문 3종의 사람 응답과 여러 LLM의 응답(전체 생성, 수정, 페르소나 기반 에이전트)을 짝지은 벤치마크. 단순한 AI 사용은 잘 잡히지만 페르소나 기반 에이전트는 chance 수준까지 떨어진다고 보고. 설문 설정의 1순위 재현 대상 |
| [Generative AI Meets Open-Ended Survey Responses](https://journals.sagepub.com/doi/10.1177/00491241251327130) (Zhang, Xu, Alvero) | Sociological Methods & Research 2025 | 없음 | 응답자의 34%가 서술형 문항에 LLM을 썼다고 답함. 사람과 LLM 응답의 차이(답변 균질화)를 분석 |
| [Detecting and managing participants' large language model use for open-ended questions in online research](https://link.springer.com/article/10.1007/s11573-026-01268-9) | 2026 | 없음 | 기존 불성실 응답·봇 탐지 지표가 LLM 사용에는 거의 통하지 않는다고 보고 |

### 참고: 한 문장 단위 탐지

- [LLM-Detector](https://arxiv.org/abs/2402.01158) (arXiv 2024, 중국어 텍스트 대상): Fast-DetectGPT 같은 기존 방법이 문장 단위 탐지에서 효과가 떨어진다고 보고. 영어에서도 그런지 재현으로 직접 확인할 가설.

## 단계

- [x] Fast-DetectGPT를 업스트림 데이터(exp_gpt3to4)로 재현. 1B급 모델로는 재현되지 않고, 원인을 분석 중 ([실험 01](experiments/01_fastdetectgpt_1b/README.md))
- [ ] Binoculars 재현, RAID 일부 데이터로 확장
- [ ] 같은 샘플을 길이별로 잘라 성능 곡선 그리기 (문단 → 3문장 → 1문장)
- [ ] 1문장 설정에서 틀린 샘플 100개를 읽고 메모
- [ ] ASURRE로 설문 설정 재현: 응답 1개만 볼 때와 한 사람의 응답 묶음을 볼 때 비교
- [ ] 패턴 정리 후 계속할지 접을지 결정
