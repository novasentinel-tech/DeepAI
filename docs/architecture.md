# 🏗️ System Architecture

## Layered Architecture Overview

```
┌─────────────────────────────────────┐
│      INPUT LAYER                    │
│  Domain/URL Validation & Whitelist  │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│   PASSIVE COLLECTION LAYER          │
│  • HTTP Headers • TLS/SSL           │
│  • DNS Records  • Tech Stack        │
│  • Port Scanning • WHOIS            │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│   FEATURE ENGINEERING LAYER         │
│  87 Features from Raw Data          │
│  • Normalization & Encoding         │
│  • Anomaly Detection                │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│   ML CORE (Supervised)              │
│  LightGBM Classification            │
│  • 4 Risk Classes                   │
│  • Probability Outputs              │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│   RL OPTIMIZATION (Prioritization)  │
│  PPO Agent Decision Making          │
│  • 10 Possible Actions              │
│  • Reward Learning                  │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│   EXPLAINABILITY LAYER              │
│  • SHAP Values                      │
│  • Natural Language Generation      │
│  • Actionable Recommendations       │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│   SECURITY & AUDIT LAYER            │
│  • Rate Limiting                    │
│  • Timeout Enforcement              │
│  • Immutable Audit Log              │
│  • Academic Mode Enforcer           │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│   OUTPUT & REPORTING                │
│  JSON Report + HTML + Audit Trail   │
└─────────────────────────────────────┘
```

## Component Interactions

### 1. Entry Point Security
- Domain validator enforces blacklist
- Rate limiter checks usage quotas
- Academic mode enforcer validates policy submission

### 2. Data Collection (Parallel)
- HTTP collector → Headers analysis
- TLS collector → Certificate & cipher inspection
- DNS collector → Record enumeration
- Tech stack detector → Fingerprinting
- Port scanner → Open services identification

### 3. Feature Engineering
- Raw data normalization (Min-Max scaling)
- Categorical encoding (one-hot)
- Feature validation & anomaly detection
- Produces consistent 87D vectors

### 4. Machine Learning Classification
- LightGBM ingests features
- Outputs class probabilities (LOW, MEDIUM, HIGH, CRITICAL)
- Feature importance extracted
- Confidence score calculated

### 5. Reinforcement Learning Prioritization
- State built from ML output + context
- PPO agent selects optimal action
- 10 possible prioritization actions
- Learning from analyst feedback

### 6. Explainability Generation
- SHAP calculates feature contributions
- NLG templates convert to human text
- Recommendations assembled
- Context-specific insights added

### 7. Security & Audit
- Rate limit incremented
- Audit log entry appended (hash-chained)
- Timeout verified
- Academic mode constraints checked

### 8. Report Generation
- JSON output assembly
- HTML report generation (optional)
- Compliance verification
- Archival with timestamp

---

## Key Design Principles

### Security First
- ✓ No exploitation capability whatsoever
- ✓ Hardened against misuse
- ✓ Immutable audit trail
- ✓ Mandatory academic mode

### Transparency & Explainability
- ✓ Every prediction explained
- ✓ All evidence cited
- ✓ Confidence quantified
- ✓ Human-readable output

### Rigorous Evaluation
- ✓ 85%+ accuracy target
- ✓ 95%+ critical recall (high priority)
- ✓ 5-fold stratified CV
- ✓ Class-weighted metrics

### Continuous Improvement
- ✓ Offline RL training
- ✓ Analyst feedback loop
- ✓ Quarterly retraining
- ✓ Model versioning

---

## Data Flow Examples

### Example 1: Benign Low-Risk Site

```
Input: google.com
  ↓
Validation: PASS ✓
  ↓
Collection: Headers ✓, TLS 1.3✓, Sec Headers ✓
  ↓
Features: [1.3, 256, True, True, False, ...]
  ↓
ML Classification: LOW (p=0.92)
  ↓
RL Action: PRIORITY_LOW
  ↓
Output: {
  "classification": "LOW",
  "confidence": 0.92,
  "priority": "LOW",
  "explanation": "Industry-standard security..."
}
```

### Example 2: Vulnerable High-Risk Site

```
Input: vulnerable-site.com
  ↓
Validation: PASS ✓
  ↓
Collection: TLS 1.0✗, No HSTS✗, Outdated CMS✗
  ↓
Features: [1.0, 128, False, True, True, ...]
  ↓
ML Classification: HIGH (p=0.87)
  ↓
RL Action: PRIORITY_CRITICAL (upgraded from HIGH)
  ↓
Output: {
  "classification": "HIGH",
  "confidence": 0.87,  
  "priority": "CRITICAL",
  "recommendations": [
    "Update TLS to 1.3...",
    "Implement HSTS...",
    "Patch CMS..."
  ]
}
```

### Example 3: Blocked Dangerous Target

```
Input: some-government-agency.gov
  ↓
Validation: BLOCKED ✗
  "Blocked TLD: .gov"
  ↓
Audit Log: BLOCKED event recorded
  ↓
Output: {
  "status": "blocked",
  "reason": "Critical infrastructure protected"
}
```

---

## File Structure Details

```
src/
├── collectors/          # Data gathering
│   ├── http_collector.py        # 10s timeout
│   ├── tls_collector.py         # 15s timeout  
│   ├── dns_collector.py         # 5s timeout
│   └── base_collector.py        # Base class
│
├── features/            # Feature creation
│   ├── feature_extractor.py     # Main extractor
│   ├── feature_definitions.py   # 87 feature specs
│   ├── normalizers.py           # Min-Max, Std scaling
│   └── validators.py            # Sanity checks
│
├── models/
│   ├── supervised/      # ML models
│   │   ├── lgbm_classifier.py   # LightGBM wrapper
│   │   ├── trainer.py           # Training pipeline
│   │   └── evaluator.py         # Metrics calculation
│   │
│   └── reinforcement/   # RL agents
│       ├── ppo_agent.py         # PPO implementation
│       ├── environment.py       # Simulation environment
│       ├── reward_function.py   # Reward logic
│       └── trainer.py           # Training loop
│
├── security/            # Security enforcement
│   ├── domain_validator.py      # Blacklist/whitelist
│   ├── rate_limiter.py          # Usage quotas
│   ├── timeout_manager.py       # Operation limits
│   ├── academic_mode.py         # Policy enforcement
│   └── audit_log.py             # Immutable logging
│
├── explainability/      # Interpretation
│   ├── shap_explainer.py        # SHAP values
│   ├── nlg_generator.py         # Human text generation
│   └── templates.py             # Explanation templates
│
└── pipeline/            # Orchestration
    ├── scan_pipeline.py         # Main workflow
    ├── analysis_pipeline.py     # Analysis steps
    └── report_generator.py      # Output formatting
```

---

## Configuration Hierarchy

```
defaults
  ↓ (overridden by)
Environment variables (.env.example)
  ↓ (overridden by)  
Runtime arguments
  ↓
Final Configuration Applied
```

## Model Versioning

```
ML Models:
  latest → v2.3.1 (current)
  ├── v2.3.0 (previous)
  ├── v2.2.0
  └── v1.0.0 (experimental)

RL Models:
  latest → v1.2.0
  └── v1.0.0

SHAP Explainers:
  latest → v1.0.0
```

---

**Last Updated: February 26, 2026**
