# System Architecture

```mermaid
flowchart LR
    subgraph Data["Data & Validation"]
        K[(Kaggle<br/>noshow.csv<br/>SHA-256 v1)] --> I[ingest]
        I --> V{Pandera<br/>schema}
        V -- fail --> X[stop flow]
        V -- pass --> S[time-based split<br/>train / val / test]
    end

    subgraph Train["Training (Prefect DAG: make pipeline)"]
        S --> F[shared preprocessing<br/>sklearn Pipeline]
        F --> M[train Exp 1-3<br/>LogReg / LightGBM]
        M --> E[evaluate]
        E --> G{model gate<br/>configs/slo.yaml}
    end

    subgraph Reg["MLflow"]
        T[(tracking<br/>params, metrics,<br/>artifacts, git/data SHA)]
        R[(registry clinic-noshow<br/>challenger / champion /<br/>previous_champion)]
    end

    M -. log .-> T
    E --> R
    G -- pass: promote --> R
    G -- fail --> D

    subgraph Serve["Serving (Docker Compose)"]
        A[FastAPI<br/>/predict /predict_batch<br/>/health /metrics]
    end

    R -- load champion / reload --> A
    U[Clinic staff /<br/>nightly batch job] --> A

    subgraph Mon["Monitoring"]
        P[Prometheus] --> GR[Grafana<br/>dashboards + alerts]
        EV[Evidently<br/>data drift] --> RT{drift or<br/>recall drop?}
        CD[recall on labels<br/>after appointment] --> RT
    end

    A -- metrics --> P
    A -- prediction logs --> EV
    RT -- yes: retrain --> I
    RB[rollback CLI] --> R

    D[Discord alerts]
    X --> D
    GR --> D
    RT --> D

    subgraph CI["GitHub Actions"]
        C1[code-quality] --> C2[data-validation] --> C3[model-gate]
    end
```

## ส่วนประกอบ

| ชั้น | เครื่องมือ | ผู้รับผิดชอบ |
|---|---|---|
| Data & Validation | pandas, Pandera, SHA-256 data version | พีรพงษ์ |
| Features & Modeling | scikit-learn Pipeline, LightGBM, SHAP | ธันว์ |
| Tracking, Registry & Pipeline | MLflow, Prefect | กิตตินันท์ |
| Serving & Infra | FastAPI, uvicorn, Docker Compose, Locust | ภูรินทร์ |
| Monitoring | Evidently, Prometheus, Grafana | ฑีฌานนท์ |
| CI/CD | GitHub Actions, ruff, pytest | ณันทพงศ์ |
| Alerting | Discord webhook | ทุกส่วน |
