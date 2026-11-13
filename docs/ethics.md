# Ethics & Responsible Use / 伦理与负责任使用

**Read this before deploying VoiceVeritas. 部署前请先阅读本文。**

VoiceVeritas detects *stylistic* evidence of reading/reciting in speech. It is a
decision-support signal — **never a verdict**. Misusing it can cause real harm
to real people.

VoiceVeritas 检测的是语音中"朗读/背诵"的**风格学**证据。它是决策辅助信号，
**不是裁决**。滥用会对真实的人造成真实伤害。

## Intended use 预期用途

- ✅ Assisting **human** reviewers of recorded interviews by highlighting
  segments that deserve a second listen.
- ✅ Research into speech style, paralinguistics, and fairness of interview
  assessment.
- ✅ Quality assurance of *practice* / mock-interview products with user consent.

## Prohibited use 禁止用途

- ❌ Fully automated rejection of candidates without human review.
- ❌ Covert monitoring — using this on people who have not been informed that
  speech-style analysis is applied.
- ❌ Use as evidence of "cheating" by itself. A high read-probability has many
  benign explanations (see below).
- ❌ Any use in jurisdictions/settings where such analysis is prohibited by law
  or employment agreements.

## Why false positives are the primary harm 为什么误报是首要风险

A fluent, well-prepared, memorized-from-rehearsal, or simply well-spoken
candidate can score "read-like": low disfluency, steady pace, smooth F0.
The tool measures **style, not honesty**:

- Speakers with certain accents, non-native speakers, and neurodivergent
  speakers may have stylistic patterns the model misreads.
- Rehearsed-but-honest answers are legitimate interview preparation.
- Measurement conditions (cheap microphones, compression, room echo) shift
  scores.

**Operating rule**: treat the score as a *prior for human attention*, choose
thresholds by **FPR@TPR95** (see [benchmark.md](benchmark.md)), and document
the false-positive rate you operate at.

## Operational safeguards 运营防护建议

1. **Human-in-the-loop**: no adverse decision on the model output alone.
2. **Calibration & threshold policy**: publish FPR at your operating point;
   re-validate on your population before deployment.
3. **Bias evaluation**: stratify error rates by accent / L2 status / gender
   where metadata allows; retrain or recalibrate if gaps appear.
4. **Privacy**: audio is processed transiently; do not retain candidate audio
   longer than necessary; comply with local privacy law (e.g. PIPL, GDPR).
5. **Transparency**: tell candidates that automated speech-style analysis may
   be used to assist review.
6. **Appeal path**: give candidates a way to contest automated flags.

## Data ethics 数据伦理

Training data comes from public research corpora (LibriSpeech, SBCSAE, …)
with documented licenses — see [data.md](data.md). We never redistribute raw
audio in this repository, and we do not train on candidate data without
explicit consent.

## Reporting misuse 报告滥用

If you become aware of a deployment that violates these principles, please
open a confidential issue at
[github.com/StanleyChanH/VoiceVeritas/issues](https://github.com/StanleyChanH/VoiceVeritas/issues).
