# 🌋 PrebioticNet: Geometric Deep Learning for Topological Chemical Evolution

PrebioticNet is an end-to-end geometric deep learning framework designed to model early-Earth chemical evolution and primitive metabolic self-assembly. Utilizing a custom Heterogeneous Graph Neural Network (HGNN) built with PyTorch Geometric, the system maps structural molecular characteristics alongside varying global environmental conditions (Temperature, pH) to predict the link-feasibility of prebiotic reactions. 

By applying network theory and directed graph traversal, PrebioticNet autonomously isolates closed, self-sustaining **autocatalytic feedback loops**—the core chemical architecture believed to be the precursor to metabolic life.

---

## 🔬 Core Architecture & ML Pipeline

Huggingface Spaces link: [text](https://huggingface.co/spaces/KdLearner/prebioticnet/blob/main/app.py)

### 1. Heterogeneous Graph Construction
The primordial soup is modeled as a directed, bipartite heterogeneous graph structure containing two distinct node classes:
* **Molecules ($V_{mol}$):** Encoded dynamically using **RDKit**. Hand-engineered feature vectors capture foundational atomic properties: heavy atom counts, valence configurations, formal charge matrices, hydrogen-bond donors, and hydrogen-bond acceptors.
* **Reactions ($V_{rxn}$):** Encoded via dynamic environmental features representing the thermodynamic environment ($[Temperature\ (K), pH]$).
* **Edges:** Formulated as directional relationships representing inputs and outputs (`molecule -> consumes -> reaction` and `reaction -> yields -> molecule`).

### 2. Self-Supervised Link Prediction
Because real-world prebiotic tracking lacks exhaustive negative data, the network optimizes via **Contrastive Negative Edge Sampling**. During training, for every authenticated reaction yield edge, the pipeline corrupts the target index to sample a structurally invalid molecular node. The HGNN passes message-passing vectors through a custom relational link predictor, optimizing via Binary Cross-Entropy with Logits Loss to map localized reaction kinetics.

### 3. Topological Cycle Discovery
Once edge affinities are inferred by the model, the graph is analyzed using a custom network traversal engine. By applying **Tarjan's Strongly Connected Components (SCC)** algorithm, the engine filters out weak structural links and isolates closed cyclic paths where a group of molecules mutually catalyze and sustain the synthesis of one another.

---

## 💻 Web Interface & Deployment

The system features a live, interactive web dashboard built with **Gradio** and hosted on **Hugging Face Spaces**. 

* **Environmental Stress Sliders:** Users can dynamically shift global surface temperatures and atmospheric pH levels.
* **Dynamic Threshold Gating:** The pipeline feeds user criteria into the model, applying an environmental boundary gate where severe deviations put structural stress on the network connectivity.
* **Live Layout Visualization:** Employs **NetworkX** and a tuned spring-repulsion physics layout to isolate, sanitize, and visually map out the surviving metabolic cycles in real-time.

---

## 🛠️ Repository Structure

```text
prebioticnet/
├── data/
│   └── chemorigins-data/     # Raw prebiotic reaction records & environmental conditions
├── src/
│   ├── dataset.py            # Heterogeneous PyG graph compiler & parsing pipeline
│   ├── features.py           # RDKit molecular feature engineering
│   ├── model.py              # Heterogeneous GNN & relational link predictor architecture
│   ├── search.py             # Network topology traversal & Tarjan's SCC cycle tracker
│   └── generative.py         # In silico hypothetical chemical path expansion engine
├── app.py                    # Gradio Web Dashboard & visualization panel
├── main.py                   # Local optimization pipeline and evaluation script
├── requirements.txt          # Package dependencies
└── LICENSE                   # MIT License
```
---

## 📊 Technical Performance & Validation

Model performance is validated right after the optimization cycle using industry-standard machine learning metrics to track link discrimination capability:

* **Classification Accuracy:** Assesses binary edge validity across positive target yields and contrastive noise.
* **ROC-AUC Score:** Evaluates the probability-ranking confidence of the model, ensuring genuine chemical transitions are cleanly segregated from structural anomalies.

---

## 🚀 Installation & Local Execution

### 1. Clone the Repository + Data
```bash
git clone https://github.com/KDLearner123/prebioticnet.git
cd prebioticnet
git clone https://github.com/brunocuevas/chemorigins-data.git
```

### 2. Configure the Environment

Ensure you have Python 3.10+ installed. It is highly recommended to use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Pipeline & App

To train the model and print the validation metrics locally:
```bash
python main.py
```

To launch the local interactive Gradio dashboard web server:
```bash
python app.py
```

## License

Distributed under the MIT License. See **LICENSE** for more information.
