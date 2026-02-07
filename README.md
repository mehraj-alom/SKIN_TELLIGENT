# SKIN_TELLIGENT

**AI Decision Support System with Confidence-Gated Inference and Explainability**

A multi-stage AI decision-support system that performs image-based skin region analysis, provides explainability signals, and conditionally enables user interaction based on model confidence — designed to prevent unreliable outputs in safety-sensitive contexts.

<p align="center">
  <img src="https://i.ibb.co/hJfvCpD3/pipeline-overview.png" alt="Detection and Classification Pipeline" width="750">
</p>

---

## 🎯 System Overview

SKIN_TELLIGENT is **not a diagnostic tool**. It is an educational decision-support system that demonstrates:

- **Multi-stage inference pipelines** with structured handoffs
- **Confidence-gated behavior** to prevent unreliable outputs
- **Explainability integration** for model transparency
- **Knowledge-restricted conversational interfaces** that operate within defined boundaries

---

## 🔐 Decision Governance & Safety

The system implements confidence-based inference gating to manage output reliability.

### Inference States

| Confidence Level | State | System Behavior |
|------------------|-------|-----------------|
| ≥ 80% | `HIGH_CONFIDENCE` | Full explainability + knowledge-restricted assistant |
| 60–80% | `UNCERTAIN` | Limited educational context, explicit uncertainty messaging |
| < 60% | `ABSTAIN` | No detailed output, professional consultation referral only |

### Safety Constraints

- **No clinical claims** — System explicitly positions all outputs as educational
- **Knowledge-restricted chat** — Conversational layer activates only after valid inference, constrained to analysis context
- **Abstention by design** — Low-confidence predictions trigger graceful degradation, not forced outputs
- **Audit trail** — All inference decisions are logged with confidence scores

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Image Input    │────▶│  YOLO Detection  │────▶│  ROI Extraction │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Grad-CAM++     │◀────│  Classification  │◀────│  Confidence     │
│  Explainability │     │  (27 classes)    │     │  Gating         │
└────────┬────────┘     └──────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Conditional Interface Layer                                     │
│  - HIGH: Full results + explainability + knowledge-restricted   │
│          conversational assistant                                │
│  - UNCERTAIN: Limited context + uncertainty messaging           │
│  - ABSTAIN: Referral message only                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Detection | YOLO v8 | Region localization |
| Classification | PyTorch (Custom CNN) | Condition identification |
| Explainability | Grad-CAM++ | Decision attribution |
| Conversational Layer | LangGraph  | State-managed, context-restricted chat |
| Interface | Streamlit | Web application |
| Containerization | Docker | Deployment consistency |

---

## 📁 Project Structure

```
SKIN_TELLIGENT/
├── src/
│   ├── streamlit_app/
│   │   ├── app.py              # Application with confidence-gated UI
│   │   └── chatbot.py          # Knowledge-restricted conversational layer
│   ├── inference/
│   │   └── pipeline.py         # Multi-stage inference pipeline
│   ├── classification/
│   │   └── classifier.py       # Classifier with confidence scoring
│   └── detection/
│       └── detector.py         # YOLO-based region detector
├── models/                     # Trained model artifacts
├── output/                     # Inference logs and results
├── requirements.txt
└── docker-compose.yml
```

---

## � Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/SKIN_TELLIGENT.git
cd SKIN_TELLIGENT
python3 -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add OPENAI_API_KEY for conversational layer

# Run application
streamlit run src/streamlit_app/app.py
```

---

## 🗺️ Roadmap

### System Maturity
- [x] Structured inference contracts with schema validation (`src/inference/contracts.py`)
- [x] Comprehensive audit logging for all model decisions (`src/inference/audit.py`)
- [x] Model version tracking and rollback capability (`src/config/model_registry.py`)
- [ ] Data drift detection and alerting
- [x] Failure-state UX with graceful degradation paths

### Evaluation & Monitoring
- [ ] Evaluation benchmarks with held-out test sets
- [ ] Confidence calibration analysis
- [ ] Monitoring hooks for production observability


---

## ⚠️ Important Disclaimer

This system is designed for **educational and research purposes only**.

- Not FDA approved
- Not clinically validated
- Not intended for medical diagnosis
- Professional medical consultation should always be sought

All outputs are explicitly framed as educational content, not medical advice.

---

## � Author

**Mehraj Alom Tapadar**

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
