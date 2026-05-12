# An Interpretable Hybrid Machine Learning and Rule-Based System for Arrival-Time Patient Triage and Hospital Resource Management

## Abstract

Timely and consistent triage at patient arrival is critical for safe and efficient hospital operations, particularly in resource-constrained environments. This paper presents the design and architecture of an interpretable hybrid decision-support system that operates from arrival-time data to assist in triaging patients, assigning departments, estimating outpatient department (OPD) waiting times, and managing bed and token-based patient flow. The system combines a Logistic Regression model for estimating patient criticality with rule-based logic for department mapping, deterioration risk estimation, and intensive care unit (ICU) versus ward suitability. Textual symptoms and procedures at arrival are encoded using term frequency–inverse document frequency (TF-IDF), while categorical variables such as gender are processed through One-Hot Encoding to avoid spurious ordinal relationships. A weighted, transparent scoring mechanism combines criticality, risk, age, and waiting time into a unified priority score, which then drives advisory recommendations regarding ICU or ward admission. A rate-limited token generation mechanism and queue-based scheduling are used to control OPD inflow and dynamically estimate waiting times, backed by a centralized state store for real-time capacity tracking. All outputs are advisory and explicitly designed for review and override by medical professionals. This work describes the design rationale, system architecture, and operational workflows, and it discusses interpretability, safety, and integration considerations for real-world deployment.

---

## 1. Introduction

Hospital emergency and outpatient departments routinely face high and variable patient inflow, heterogeneous case mix, and constrained resources. In such contexts, small delays or misjudgments in triage can translate into significant clinical risk and operational inefficiencies. Traditional triage protocols rely heavily on human expertise and rule-based checklists, which may be inconsistently applied under pressure, subject to fatigue, and limited in their ability to integrate real-time capacity and queuing information.

Recent advances in machine learning (ML) have enabled more data-driven risk estimation, yet black-box models can be difficult to interpret, validate, and govern in safety-critical domains such as healthcare. Regulatory and ethical expectations increasingly emphasize explainability, auditability, and human oversight.

This paper presents a hybrid, interpretable system that assists clinicians and administrators from the earliest available information point: patient arrival. The system:

- Ingests basic structured and textual patient information available at arrival time.
- Estimates patient criticality using an interpretable Logistic Regression model.
- Assigns patients to hospital departments via deterministic rule-based mappings.
- Estimates OPD waiting time based on dynamic queuing and current patient load.
- Manages OPD tokens with rate-limiting to prevent overload.
- Tracks ICU and ward bed availability in real time via centralized state.
- Computes a transparent priority score that combines criticality, deterioration risk, age, and waiting time.
- Suggests ICU or ward suitability using threshold-based rules, while maintaining clinician override.
- Updates hospital state after every patient interaction for consistency.

The focus of this work is on system design and algorithmic choices that preserve interpretability, safety, and operational realism, rather than on maximizing predictive accuracy via complex models.

---

## 2. Related Work

Triage and patient flow management have been studied extensively in the emergency medicine and operations research communities. Common approaches include:

- **Rule-based triage scales** such as the Emergency Severity Index (ESI) and the Manchester Triage System, which use predefined criteria to categorize urgency.
- **Queueing-theoretic models** to estimate waiting times and optimize staffing levels.
- **Machine learning–based risk prediction** models for mortality, ICU transfer, or readmission, often using supervised learning with electronic health record (EHR) data.

While ML-based triage tools have shown improvements in discrimination and calibration in some settings, many adopt complex models (e.g., gradient boosting, deep learning) that may be difficult to interpret and validate clinically. At the same time, operations-focused systems often model waiting times and bed utilization, but are not tightly integrated with clinical risk estimation at arrival.

This work integrates three strands—arrival-time risk prediction, rule-based clinical logic, and dynamic resource-aware scheduling—into a single interpretable, operationally grounded system intended for advisory use.

---

## 3. System Overview

The proposed system operates as a decision-support layer on top of existing hospital information systems. At a high level, the system workflow is:

1. **Arrival-time data intake**
   - Basic demographics (e.g., age, gender).
   - Presenting complaints and symptoms (free text).
   - Planned or recent procedures (free text, if available).
   - Vital signs or high-level condition flags if captured at registration.

2. **Feature processing**
   - TF-IDF vectorization of symptom/procedure text.
   - One-Hot Encoding of categorical attributes.
   - Concatenation with numeric features (e.g., age, vital sign summaries).

3. **Machine learning prediction**
   - Logistic Regression model to estimate a criticality probability (e.g., probability of needing ICU-level care or rapid intervention).

4. **Rule-based clinical logic**
   - Department assignment via keyword and condition mapping (e.g., cardiology, neurology, surgery).
   - Deterioration risk estimation from critical symptom/procedure keywords and flags.

5. **Operational state and queuing**
   - Real-time tracking of ICU and ward bed availability.
   - Token-based OPD flow control with rate-limited token issuance.
   - Queue-based waiting time estimation for OPD services.

6. **Priority scoring and recommendations**
   - Computation of a priority score from criticality, risk, age, and waiting time.
   - Threshold-based decision rules for ICU vs. ward suitability.

7. **State update and logging**
   - Update of hospital capacity and queue state after each patient interaction.
   - Logging of explanations and feature contributions for auditability.

All outputs—criticality estimates, department assignments, waiting times, and ICU/ward recommendations—are advisory. Final decisions remain with medical professionals.

---

## 4. Data and Feature Engineering

### 4.1 Arrival-Time Data Inputs

The system assumes only information that is realistic to collect at or near arrival, including:

- Age (numeric).
- Gender and possibly pregnancy status (categorical).
- Symptom description (short free text, e.g., "chest pain, shortness of breath").
- Known conditions or comorbidities if reported (e.g., "diabetes, hypertension").
- Recent or planned procedures if documented at registration.
- Optional: coarse triage flags (e.g., "trauma", "unconscious", "bleeding").

This constraint ensures feasibility and minimizes clinician and registration workload.

### 4.2 Text Feature Extraction via TF-IDF

Symptom and procedure descriptions are transformed into numerical features using TF-IDF vectorization. Let the corpus of arrival notes be {d₁, …, dₙ}. For each term t in the vocabulary:

- Term frequency tf(t, d) is the count (or normalized count) of t in document d.
- Inverse document frequency idf(t) = log(N / (1 + nₜ)), where nₜ is the number of documents containing t.

The TF-IDF weight is tfidf(t, d) = tf(t, d) · idf(t).

TF-IDF is chosen because arrival-time text is typically short, with keyword presence and absence more informative than deep semantic structure. This representation is:

- Sparse and efficient.
- Highly interpretable: individual term weights can be inspected.
- Compatible with linear models such as Logistic Regression.

### 4.3 Categorical Feature Encoding via One-Hot Encoding

Categorical variables such as gender (e.g., male, female, other) and possibly arrival mode (e.g., walk-in, ambulance) are encoded using One-Hot Encoding. For a categorical feature with k categories, this yields a k-dimensional binary vector, with exactly one element set to 1.

This avoids imposing arbitrary ordinal relationships (e.g., encoding "male" as 0 and "female" as 1), which could otherwise introduce spurious linear structure in the model.

### 4.4 Numeric Features

Numeric features such as age and derived statistics from vital signs (if available) are included directly, optionally after normalization or standardization (e.g., z-score scaling). These features are straightforward to interpret within a linear model and are important components of risk.

---

## 5. Machine Learning Component: Logistic Regression

### 5.1 Problem Formulation

The system uses a binary Logistic Regression model to estimate patient criticality at arrival. The target label can be defined in various clinically meaningful ways, for example:

- Need for ICU-level care within a specified time horizon.
- Need for rapid intervention (e.g., surgery, invasive ventilation).
- High mortality risk within a defined period.

Given a feature vector x ∈ ℝᵈ derived as described above, the model estimates:

P(y = 1 | x) = σ(w^T x + b)

where y = 1 denotes "critical" (according to the chosen definition), w ∈ ℝᵈ is the weight vector, b ∈ ℝ is the bias term, and σ(z) is the logistic function σ(z) = 1 / (1 + e^(-z)).

### 5.2 Training Objective

The model is trained by minimizing the regularized logistic loss:

ℒ(w, b) = -Σᵢ₌₁ᴺ [yᵢ log ŷᵢ + (1 - yᵢ) log(1 - ŷᵢ)] + λ ||w||₂²

where ŷᵢ = P(yᵢ = 1 | xᵢ) and λ is an L₂ regularization parameter to prevent overfitting and improve stability.

### 5.3 Rationale for Logistic Regression

Logistic Regression is chosen over more complex models for the following reasons:

- **Speed**: Inference and training are fast, suitable for high-throughput environments.
- **Stability**: The convex objective and mature optimization methods yield stable and reproducible models.
- **Interpretability**: Each feature's coefficient directly indicates its contribution to log-odds of criticality, enabling clinicians to examine which symptoms, procedures, or demographics drive risk estimates.
- **Auditability**: Linear models are amenable to regulatory review and model governance, and they integrate well with post-hoc explanation methods if needed.

The model's probability output is used as an input to the downstream priority scoring and ICU/ward decision logic, not as a sole decision driver.

---

## 6. Rule-Based Clinical and Operational Logic

### 6.1 Department Assignment

Patients are assigned to hospital departments via a deterministic, rule-based logic grounded in keyword matching and condition flags. For example:

- Symptoms including "chest pain", "shortness of breath", or procedures like "angioplasty" map to Cardiology.
- Symptoms including "seizure" or "loss of consciousness" may map to Neurology.
- Trauma-related keywords (e.g., "fracture", "road traffic accident") may map to Orthopedics or General Surgery.

The design goal is:

- **Determinism and auditability**: Given the same input, the mapping is consistent and easily explained.
- **Configurable mapping tables**: Clinical administrators can update mapping rules as clinical pathways evolve.
- **Fallback handling**: For ambiguous or multi-domain symptoms, the system can either assign a default department or flag for manual routing.

### 6.2 Deterioration Risk Estimation

In addition to ML-based criticality, a rule-based deterioration risk score is computed from high-risk condition and procedure keywords. For example:

- Any mention of "STEMI", "cardiogenic shock", or "massive GI bleed" might trigger a high-risk flag.
- Procedures like "emergency laparotomy" or "intubation" are treated as critical.

The design principle is conservative: critical procedures and canonical high-risk indicators are never allowed to be underweighted by the data-driven model. This redundancy supports safety by ensuring that obvious red flags translate into high deterioration risk, independent of the Logistic Regression output.

---

## 7. Priority Scoring and ICU/Ward Recommendation

### 7.1 Priority Score Composition

A composite priority score S is computed from four main components:

- C: ML-estimated criticality probability.
- R: rule-based deterioration risk score (e.g., normalized in [0, 1]).
- A: age factor (to represent vulnerability, if clinically appropriate).
- W: normalized waiting time or time since arrival.

A simple linear scoring function is used:

S = wc·C + wR·R + wA·A + wW·W

where wc, wR, wA, wW are non-negative weights chosen in consultation with clinicians and operations experts. This form is:

- **Transparent**: Contributions from each component are explicit and explainable.
- **Configurable**: Weights can be tuned to reflect local policy and constraints (e.g., giving higher weight to waiting time in crowded OPDs).
- **Monotonic**: Increasing any individual component increases the overall priority.

### 7.2 ICU vs. Ward Threshold Rules

The ICU versus ward suitability recommendation is determined via threshold-based rules:

- If S ≥ θ_ICU, the system recommends ICU suitability (subject to bed availability).
- If θ_ward ≤ S < θ_ICU, the system recommends ward admission.
- If S < θ_ward, the system may recommend standard OPD follow-up.

Thresholds θ_ICU and θ_ward are calibrated to reflect:

- Institutional policies and risk tolerance.
- Historical triage outcomes if data are available.
- Capacity constraints and practical considerations.

The recommendations are advisory: clinicians can override them, and every override is logged to support governance and potential retraining or rule adjustment.

---

## 8. OPD Flow Control and Waiting Time Estimation

### 8.1 Token-Based Inflow Control

The system uses a rate-limited token generation algorithm to manage OPD patient inflow:

- Each new arrival requesting OPD services is issued an electronic token based on current load and service capacity.
- A sliding time window and maximum tokens-per-window parameter enforce a ceiling on how many new patients can enter the OPD queue within a given interval.
- In periods of high inflow, token issuance slows or temporarily pauses, and patients can be informed of extended waiting times or advised to seek alternative services if clinically appropriate.

This mechanism prevents OPD overload, which could otherwise lead to excessive waiting times and degraded care quality.

### 8.2 Queue-Based Waiting Time Estimation

The system models OPD waiting time using explicit queues:

- For each OPD service or physician, a queue of tokens (i.e., patients) is maintained.
- Estimated waiting time for a new patient is computed from:
  - Current queue length.
  - Historical or configured average service time per patient.
  - Service level agreements (e.g., target maximum waiting times).

Dynamic updates occur whenever:

- A patient is seen, no-shows, or leaves.
- Capacity changes (e.g., additional staff or consulting rooms come online).

This queue-based model aligns with real-world service behavior, is intuitive to clinicians and administrators, and can be tuned using historical data.

---

## 9. Centralized State Management

A centralized state-tracking mechanism maintains a consistent, real-time view of:

- **Bed capacity**:
  - ICU beds: total, occupied, reserved, and available.
  - Ward beds by type or specialty.
- **OPD capacity**:
  - Active queues per service/physician.
  - Token counts and throughput statistics.
- **Pending decisions and follow-ups**:
  - Patients waiting for admission decisions or transfers.

All modules—ML prediction, rule-based logic, token manager, and queue scheduler—interact with this centralized state, ensuring consistency and avoiding conflicting decisions. Every patient interaction (arrival, triage decision, admission, discharge) triggers a synchronous state update and generates an auditable event log entry.

---

## 10. Interpretability, Safety, and Governance

### 10.1 Interpretability Features

The system is intentionally designed for interpretability:

- Logistic Regression provides per-feature coefficients that can be inspected and summarized.
- TF-IDF allows clinicians to see which symptom and procedure terms most strongly influence risk estimates.
- Rule-based department mapping and deterioration risk logic are expressed as explicit tables and rules.
- The priority score is decomposable into contributions from criticality, risk, age, and waiting time.

Explanations can be rendered to users (e.g., "High ICU suitability because: high criticality score from 'shortness of breath, chest pain', high deterioration risk from 'STEMI', long waiting time").

### 10.2 Safety and Human Oversight

Key safety mechanisms include:

- **Advisory-only outputs**: The system does not execute automatic admissions or transfers; clinicians retain full authority.
- **Conservative rule-based overrides**: High-risk keywords ensure critical cases are prioritized even if the model underestimates their probability.
- **Override logging**: Any divergence between system recommendation and clinician decision is recorded, supporting later analysis.
- **Configuration and governance**:
  - Thresholds, weights, and rules are configurable under appropriate governance.
  - Changes are versioned and auditable.

### 10.3 Data Privacy and Security Considerations

While not detailed in full here, deployment must comply with relevant health data protection regulations (e.g., HIPAA, GDPR, or local equivalents), including:

- Secure storage and transmission of patient data.
- Role-based access control.
- Audit logs for all user actions.

---

## 11. Evaluation Framework

This paper focuses on design and architecture; however, rigorous evaluation is essential before clinical deployment. A multi-phase evaluation strategy may include:

1. **Retrospective validation**
   - Train and test the Logistic Regression model on historical arrival and outcome data.
   - Assess discrimination (e.g., AUC-ROC, precision-recall), calibration, and subgroup performance.

2. **Simulation-based operational evaluation**
   - Use historical inflow data to simulate OPD queues and bed occupancy under different token and threshold configurations.
   - Measure metrics such as average waiting time, proportion of patients exceeding target wait, and bed utilization rates.

3. **Prospective observational study**
   - Deploy the system in advisory mode without changing clinical protocols.
   - Monitor agreement between system recommendations and clinician decisions, along with any observed impact on process measures (e.g., time-to-ICU, time-to-consult).

4. **Controlled rollout**
   - After safety review, gradually integrate recommendations into operational decision-making in a controlled, ethically approved manner.
   - Continuously monitor for unintended consequences, bias, and alert fatigue.

Comprehensive evaluation should involve multidisciplinary stakeholders, including clinicians, operations managers, data scientists, and compliance officers.

---

## 12. Discussion

The proposed system demonstrates how a hybrid of simple, interpretable ML and rule-based logic can support arrival-time patient triage and resource-aware flow management. The most significant advantages are:

- **Interpretability and trust**: Each component can be reasoned about by domain experts.
- **Modularity**: Individual modules (e.g., department mapping, risk rules) can be updated independently.
- **Operational realism**: Token-based control and queue-based waiting time estimation reflect actual service behavior.

However, there are important limitations:

- Logistic Regression may underperform more complex models for certain prediction tasks, especially with rich nonlinear relationships.
- TF-IDF ignores deep semantic context and may be brittle to variations in phrasing, spelling, and language.
- Rule-based logic requires ongoing maintenance and may not generalize to new clinical scenarios.
- The system's effectiveness depends heavily on data quality at arrival, which may be inconsistent.

Future enhancements could include:

- Incorporating additional structured data (e.g., vital sign time series) via appropriate feature engineering.
- Exploring more advanced but still interpretable models (e.g., generalized additive models, monotonic gradient boosting with explainability constraints).
- Integrating natural language processing techniques for better understanding of free text while preserving explainability.
- Providing adaptive, data-driven suggestions for adjusting thresholds and weights based on observed outcomes.

---

## 13. Conclusion

This paper presents an interpretable, hybrid machine learning and rule-based system designed to support arrival-time triage and resource management in hospitals. By combining a Logistic Regression model using TF-IDF and One-Hot Encoded features with deterministic clinical rules, token-based OPD flow control, queue-based waiting time estimation, and centralized state management, the system delivers transparent and auditable advisory outputs on criticality, department assignment, waiting time, and ICU versus ward suitability. Its architecture is intended to align with real-world operational constraints and governance requirements, making it a pragmatic candidate for deployment in settings where safety, interpretability, and clinician oversight are paramount.

---

## References

[1] Gilboy, N., Tanabe, P., Travers, D., & Rosenau, A. M. (2012). Emergency Severity Index (ESI): A triage tool for emergency department care (Version 4). AHRQ Publication No. 12-0014. Agency for Healthcare Research and Quality.

[2] Mackway-Jones, K., Marsden, J., & Windle, J. (Eds.). (2005). Emergency triage: Manchester Triage System (2nd ed.). BMJ Publishing Group.

[3] Churpek, M. M., Yuen, T. C., Winslow, C., Meltzer, D. O., Kattan, M. W., & Edelson, D. P. (2016). Multicenter comparison of machine learning methods and conventional regression for predicting clinical deterioration on the ward. Critical Care Medicine, 44(2), 368–374.

[4] Rajkomar, A., Oren, E., Chen, K., Dai, A. M., Hajaj, N., Liu, P. J., ... & Dean, J. (2018). Scalable and accurate deep learning with electronic health records. NPJ Digital Medicine, 1(1), 18.

[5] Beam, A. L., & Kohane, I. S. (2018). Big data and machine learning in health care. Journal of the American Medical Association, 319(13), 1317–1318.

[6] Caruana, R., Lou, Y., Gehrke, J., Koch, P., Sturm, M., & Elhadad, N. (2015). Intelligible models for healthcare. In Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 1721–1730).

[7] Sutton, R. T., Pincock, D., Bevans, M., Connis, R. T., & Srivastava, R. (2020). An open letter to the FDA and CDC on artificial intelligence bias in radiology. Nature Medicine, 26(1), 12–14.