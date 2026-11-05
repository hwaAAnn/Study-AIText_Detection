# 실험 01: Fast-DetectGPT를 1B급 채점 모델로 재현

**질문:** Fast-DetectGPT가 1B급 대표 모델로도 재현되는가? 안 된다면 왜인가?

**현재 답 (2026-11-06):**
- 코드는 논문 수치를 재현한다(GPT-Neo-2.7B).
- 그런데 Llama-3.2를 채점 모델로 쓰면 GPT-3.5/4 글에서 Fast-DetectGPT가 무너지고, 같은 모델의 단순 Likelihood보다도 낮아진다. 원인은 entropy가 likelihood 상승을 상쇄하는 것이다.
- 이 상쇄는 학습 중에 LLM이 만든 글이나 분포에 노출된 모델에서 크다. 증류 모델(Llama-3.2, Gemma-3)과, 증류 없이 2023년 이후 웹·합성 데이터로 학습한 SmolLM2가 여기에 해당한다. 대부분 ChatGPT 이전 데이터로 학습한 GPT-Neo와 OLMo-2에서는 작다.

## 설정

- **데이터:** 업스트림 [`exp_gpt3to4`](https://github.com/baoguangsheng/fast-detect-gpt/tree/971b05202bac2bb504d60c0ac0812fea7a8f7c82/exp_gpt3to4)
  - 9세트: XSum(BBC 뉴스) / WritingPrompts(소설) / PubMedQA(의학 QA) × GPT-3 davinci / GPT-3.5-turbo / GPT-4
  - 세트마다 사람 글 150개와 AI 글 150개
  - AI 글은 사람 글의 앞부분(중앙값 약 23단어)을 이어 쓴 것이라, 쌍마다 앞부분이 같다. PubMed는 질문을 공유한다.
  - 생성 temperature는 0.8이다.
  - 글 길이는 XSum·Writing 약 200토큰, PubMed 약 65토큰이다.
- **점수:** Fast-DetectGPT(analytic, sampling 모델 = scoring 모델), Likelihood, LogRank, Entropy. 계산식은 업스트림 `fast_detect_gpt.py`, `baselines.py`와 같다.
- **비교 기준:** 업스트림 `exp_gpt3to4/results`의 논문 수치. Fast-DetectGPT는 GPT-J-6B로 샘플링하고 GPT-Neo-2.7B로 채점했고, 기준 방법은 GPT-Neo-2.7B를 썼다.
- **장비:** GTX 1060 6GB. 6GB를 넘는 모델은 일부를 CPU로 넘기거나(`--max_gpu_mem`), 디코더 층을 하나씩 GPU에 올려 정확한 fp16으로 돌린다(`--layerwise`, [layerwise.py](layerwise.py)). 층별 방식은 Llama-3.2-1B의 일반 fp16 결과와 소수 다섯째 자리까지 같다(20쌍, 차이 0). 4bit 디코더(`--split_4bit`)는 결과를 바꿔서 쓰지 않는다(결과 5).

## 실행

```bash
bash experiments/01_fastdetectgpt_1b/get_data.sh                       # 데이터와 논문 수치 받기 (data/, gitignore)
python experiments/01_fastdetectgpt_1b/run.py --model meta-llama/Llama-3.2-1B
python experiments/01_fastdetectgpt_1b/run.py --model EleutherAI/gpt-neo-2.7B --max_gpu_mem 4GiB
python experiments/01_fastdetectgpt_1b/run.py --model google/gemma-3-1b-pt --dtype float32 --max_gpu_mem 3500MiB
python experiments/01_fastdetectgpt_1b/run.py --model meta-llama/Llama-3.1-8B --layerwise     # 정확한 fp16, 층을 하나씩 GPU에
python experiments/01_fastdetectgpt_1b/run.py --model <model> --save_tokens --datasets xsum_davinci xsum_gpt-4
python experiments/01_fastdetectgpt_1b/analyze_scorers.py   # 채점 모델 비교표
python experiments/01_fastdetectgpt_1b/analyze_tokens.py    # 토큰 단위 분석 (L–E coupling β 등)
python experiments/01_fastdetectgpt_1b/analyze_quant.py     # 4bit 양자화 대조
```

결과는 `outputs/01_fastdetectgpt_1b/<모델>/`에 저장된다(gitignore). 4bit 실행은 `<모델>-nf4split/`, 토큰 단위 실행은 `<모델>/tokens/`에 따로 저장된다.

## 결과

### 1. 코드 검증: GPT-Neo-2.7B로 논문 수치 재현

- Likelihood, LogRank, Entropy는 9세트 × 3점수, 27칸 모두 논문과 AUROC 0.001 이내로 같다.
- Fast-DetectGPT를 Neo-2.7B 하나로 계산하면 평균 0.906이고, 논문(GPT-J/Neo 조합)은 0.913이다. 샘플링 모델과 채점 모델을 나눈 효과는 0.007 정도다.

### 2. Llama-3.2-1B에서는 Fast-DetectGPT가 무너진다

- 평균 AUROC는 0.793이다. 논문(0.913)보다 낮고, 같은 모델의 Likelihood(0.826)보다도 낮다. xsum_gpt-4에서는 0.594다.
- 샘플링 모델과 채점 모델이 같으면 Fast-DetectGPT의 분자는 Σ(log p + H), 곧 토큰 수 × (Likelihood + Entropy)다. 이 관계는 Fast-DetectGPT 논문의 식 5, When Less is More의 부록 A와 같다.
- GPT-3.5/4 글에서는 Llama의 entropy가 likelihood 상승만큼 떨어져서 분자 차이가 거의 사라진다.

| xsum_gpt-4 (AI 평균 − 사람 평균) | Llama-3.2-1B | GPT-Neo-2.7B |
|---|---|---|
| ΔLikelihood | +0.324 | +0.318 |
| ΔEntropy | −0.280 | −0.061 |
| 분자에 남은 비율 (ΔL+ΔH)/ΔL | 14% | 81% |
| Fast-DetectGPT AUROC | 0.594 | 0.885 |

- davinci(채팅 튜닝 전 GPT-3) 글에서는 Llama도 ΔEntropy가 0에 가깝다. 그래서 이 글들에서는 Fast-DetectGPT가 Likelihood보다 낫다(XSum 0.817 대 0.667).

### 3. 채점 모델 비교

"GPT-3.5/4 글" 열은 davinci를 뺀 6세트의 평균이다. 남은 비율은 6세트 평균 (ΔL+ΔH)/ΔL이다.

| 채점 모델 | 학습 데이터 | 증류 | FDG / Likelihood (GPT-3.5/4 글) | FDG < Likelihood인 세트 (6개 중) | 남은 비율 | xsum_gpt-4 FDG (남은 비율) |
|---|---|---|---|---|---|---|
| GPT-Neo-1.3B | 2020 (Pile) | 없음 | 0.928 / 0.882 | 0 | 64% | 0.888 (72%) |
| GPT-Neo-2.7B | 2020 (Pile) | 없음 | 0.932 / 0.879 | 0 | 68% | 0.885 (81%) |
| OLMo-2-0425-1B | 대부분 2023년 이전 웹 (DCLM) | 보고 없음 | 0.924 / 0.892 | 1 | 75% | 0.869 (100%) |
| SmolLM2-1.7B | 2024년까지 웹(FineWeb-Edu) + DCLM + 합성 교재(Cosmopedia v2, 4단계 4%) | 없음 (논문 확인) | 0.866 / 0.926 | 6 | 39% | 0.818 (49%) |
| Gemma-3-1B (fp32) | 2024~25 | 있음 | 0.913 / 0.929 | 5 | 54% | 0.845 (56%) |
| Llama-3.2-1B | ~2023.12 | 있음 (Llama-3.1 logits) | 0.800 / 0.904 | 6 | 33% | 0.594 (14%) |
| Llama-3.2-3B | ~2023.12 | 있음 (Llama-3.1 logits) | 0.795 / 0.911 | 6 | 30% | 0.630 (21%) |

- **크기는 원인이 아니다.** Neo는 1.3B와 2.7B 모두 정상이고, Llama는 1B와 3B 모두 무너진다.
- **정규화(σ로 나누기)도 원인이 아니다.** 나누기 전 점수(EGS)의 AUROC가 Fast-DetectGPT와 0.004 이내로 같다. 상쇄는 분자에서 일어난다.
- **최신 구조나 토크나이저 자체도 원인이 아니다.** OLMo-2(2025년 모델, 어휘 10만)는 정상이다.
- **증류가 꼭 필요하지는 않다.** SmolLM2는 증류 없이 학습했는데도 상쇄가 Llama 수준이다(남은 비율 39%, 6세트 모두 FDG < Likelihood).
- **상쇄가 큰 모델의 공통점은 LLM 산출물에 노출된 학습이다.** 증류는 큰 모델의 분포를 직접 배우는 경로다. SmolLM2는 2023년 이후 웹, LLM 합성 교재, 그리고 Llama-3-70B의 평가로 고른 웹 데이터(FineWeb-Edu)를 학습했다.
- **상쇄가 작은 모델은 대부분 ChatGPT 이전 데이터로 학습했다.** Neo와 OLMo-2다. OLMo-2 데이터의 대부분은 2023년 이전 Common Crawl(DCLM)이다.
- **어떤 노출 경로가 효과를 내는지는 아직 분리되지 않았다.** 증류가 효과를 더할 수는 있다. xsum_gpt-4에서 남은 비율은 Llama 14%, SmolLM2 49%다.

### 4. 토큰 단위 분석 (xsum_gpt-4, 사람이 쓴 공통 앞부분 제외)

`analyze_tokens.py` 결과다. L = −log p(실제 토큰), E = entropy다. β는 When Less is More 6.2절과 같은 방식으로 구한 L–E coupling 기울기다. 각 칸은 사람 글 / AI 글 순서다.

| 채점 모델 | β | 평균 E | E < 0.5인 자리 비율 | 저entropy 하위 20%의 평균 (L − E) |
|---|---|---|---|---|
| GPT-Neo-1.3B | 0.965 / 0.893 | 2.736 / 2.617 | 9.8% / 13.4% | −0.003 / −0.034 |
| GPT-Neo-2.7B | 0.957 / 0.884 | 2.613 / 2.538 | 10.9% / 14.0% | +0.027 / −0.028 |
| OLMo-2-0425-1B | 0.966 / 0.893 | 2.482 / 2.473 | 11.6% / 13.4% | +0.029 / −0.038 |
| SmolLM2-1.7B | 0.972 / 0.881 | 2.352 / 2.136 | 13.5% / 18.2% | −0.003 / **+0.011** |
| Gemma-3-1B | 1.054 / 0.977 | 2.153 / 1.950 | 13.9% / 18.1% | +0.126 / +0.066 |
| Llama-3.2-1B | 0.969 / 0.905 | 2.524 / 2.192 | 11.3% / 16.6% | −0.026 / **+0.056** |
| Llama-3.2-3B | 0.969 / 0.898 | 2.265 / 1.995 | 13.9% / 18.9% | −0.012 / **+0.063** |

- **β로는 채점 모델이 구분되지 않는다.** AI 글의 β는 Neo, OLMo, Llama 모두 0.88~0.91이다. 따라서 상쇄는 When Less is More가 말한 "글 안에서 L과 E가 따로 노는 현상(β < 1)"과 다르다. AI 글 전체의 평균 entropy가 사람 글보다 낮아지는, 집단 간 이동이다.
- **AI 글의 평균 entropy는 Llama와 Gemma에서만 크게 낮아진다.** Llama-1B는 −0.33, Llama-3B는 −0.27, Gemma는 −0.20이다. Neo와 OLMo는 −0.01~−0.12다. Llama와 Gemma에서는 거의 확실한 자리(E < 0.5)도 4~5%p 늘어난다.
- **Llama에서는 낮은 entropy 자리의 신호가 뒤집힌다.**
  - Neo와 OLMo에서는 저entropy 자리에서 AI 글의 (L − E)가 사람 글보다 낮다. 기대보다 확률 높은 토큰을 썼다는 뜻이라 정상 방향이다.
  - Llama에서는 AI 글 쪽이 더 높다. 사람 글 쪽으로 넘어갔다는 뜻이다.
  - When Less is More는 이런 "해로운 저entropy 토큰"을 약한 생성 모델에서만 보고했다. 여기서는 강한 생성 모델(GPT-4) 글에서도 최신 채점 모델로 재면 나타난다. 저entropy 토큰을 걸러내면 Llama에서 회복되는지가 다음 확인 거리다.
- davinci 글에서는 Llama의 평균 E도 사람 글과 같다(2.526 / 2.543). 상쇄는 채팅 모델 글에서만 생긴다.

### 5. 4bit 양자화 대조: 쓰지 않기로 함

Llama-3.2-1B를 fp16과 4bit 디코더(`--split_4bit`)로 같은 조건에서 비교했다(`analyze_quant.py`).
- 통제 조건: 같은 토크나이저와 토큰 ID, 같은 전체 텍스트, 같은 점수 함수, fp32 log-softmax. 패딩과 샘플링은 없다. 출력층은 두 쪽 모두 fp16이다.

| 비교 | 결과 |
|---|---|
| 토큰별 L 차이 | 평균 0.30 nat(평균 L의 약 12%), 상관 0.98 |
| 토큰별 E 차이 | 평균 0.27 nat, 상관 0.97 |
| β | 사람 글과 AI 글 모두 약 0.025 낮아짐 |
| 글별 FDG 상관 | 0.91~0.98 |
| xsum_gpt-4 FDG AUROC | 0.594 → 0.736 (남은 비율 14% → 35%) |

- 4bit 모델은 전체적으로 덜 확신한다(entropy 상승). 그 상승이 AI 글에서 더 크다(xsum_gpt-4: 사람 +0.155, AI +0.210).
- 그래서 우리가 재는 성질인 "채팅 모델 글에서의 낮은 entropy"가 일부 지워진다.
- 따라서 4bit 결과는 fp16 모델들과 비교하지 않는다. 8B는 층별 fp16 방식(`--layerwise`)으로 돌린다.
- 이 현상 자체도 단서다. 채팅 모델 글에 대한 확신은 양자화에 약한, 세밀한 가중치에 담겨 있을 수 있다.

### 6. 진행 중 (결과가 나오면 추가)

| 실험 | 확인할 것 |
|---|---|
| Llama-3.1-8B (fp16 층별) | Llama-3.2의 증류 교사 모델이다. 같은 Meta 데이터 계열에서 증류만 빠진다. 상쇄가 생기면 웹에 섞인 LLM 글만으로도 충분하다는 쪽이고, 안 생기면 증류나 합성 데이터처럼 더 직접적인 노출이 필요하다는 쪽이다 |

## 한계

- 데이터는 2023년 생성 모델이 뉴스 앞부분을 이어 쓴 글이라, 실제 채팅 응답이나 설문 응답과 다르다.
- 채점 모델 비교는 상관관계 증거다. 모델마다 데이터, 크기, 토크나이저, 학습법이 함께 다르다.
- Gemma-3는 fp16에서 수치가 넘치는 문제가 알려져 있어 fp32로 돌렸다. 나머지는 fp16이다.
- 짧은 텍스트(단일 문장, 설문 응답) 설정은 아직 다루지 않았다.

논의 과정과 선행연구 대조는 [docs/research-log.md](../../docs/research-log.md)에 있다.
