# HackVerse

A collection of AI-integrated backend and security tooling prototypes — projects that lean on LLM APIs and rule-based scoring to make backend systems a bit smarter than static logic.

## Projects

### 🛡️ NHI-Agent — NHI Governance Agent
**Security agent for non-human identity risk in code repos**

Scans GitHub/GitLab repositories for hardcoded credentials and secrets, sends findings to the Gemini 2.0 Flash API for risk scoring, and can auto-remediate by opening pull requests. Built solo for the Google Cloud Rapid Agent Hackathon 2026.

`Python` `Gemini API` `GitHub/GitLab APIs` `Risk Scoring`

### 🔍 ShadowFix — Data Pipeline Validator
**Catches data pipeline issues before they become production incidents**

Validates data pipelines end-to-end using BigQuery and a FastAPI backend, applying a weighted risk-scoring formula (rule-based, not a trained model) to flag problems before they surface downstream.

`Python` `FastAPI` `BigQuery` `Risk Scoring`

## Stack

`Python` `FastAPI` `Gemini API` `BigQuery` `Risk Scoring`

## Note

These are working prototypes focused on doing something real in a backend, not polished demos. "Intelligence" here means LLM API calls and rule-based scoring — not custom-trained ML/DL/NLP models. More projects get added as they move from private experiments to stable versions.

## License

MIT — see [LICENSE](LICENSE).
