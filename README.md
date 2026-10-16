# 🧠 Hierarchical Memory
> **Autonomous 8-level hierarchical long-term memory for local LLMs with homeostatic consolidation, hybrid retrieval, epistemic provenance, and ZERO external dependencies.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dependencies](https://img.shields.io/badge/Dependencies-0%20(Standard%20Library)-brightgreen.svg)]()
[![Backend](https://img.shields.io/badge/LLM-Local%20(OpenAI--Compatible)-orange.svg)]()

Unlike traditional vector-only RAG databases, **Hierarchical Memory** is a bio-inspired cognitive architecture. It models persistent long-term neocortical knowledge as an **8-level deep semantic ontology** and runs an autonomous **homeostatic background loop ("sleep cycle")** that continuously consolidates, forgets, merges, repairs, and self-tests memories while idle.

---

## ✨ Key Highlights

- **8-Level Hierarchical Ontology:** Organizes facts from universal domains down to atomic cells:
  Planeta ➜ País ➜ Ciudad ➜ Fábrica ➜ Persona ➜ Órgano ➜ Tejido ➜ Célula
- **Zero External Dependencies:** Built **100% on Python Standard Library** (http.server, threading, urllib, json). Zero pip install required.
- **Homeostatic Consolidation (Sleep Cycle):** While the agent is idle, an autonomous background loop runs:
  - selftest: Formulates hypothetical questions to probe its own tree and validates recall paths.
  - decay: Archives low-importance memories using an exponential half-life curve.
  - dedupe & merge: Identifies and fuses redundant facts and sibling branches.
  - repair: Prunes orphan branches and rebuilds lexical indices.
- **Hybrid Multi-Method Retrieval:** Combines exact match, BM25 lexical ranking, LLM semantic query expansion, and tree beam-search, fused via **Reciprocal Rank Fusion (RRF)**.
- **Epistemic Provenance (Source Tracking):** Automatically attributes every ingested fact to its original paper, textbook, author, or timestamped conversation to prevent hallucinations.
- **Native Real-Time Web Console:** Complete dark-mode dashboard with visual cognitive pipelines, tree rendering, and instant loop pause/resume.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Any OpenAI-compatible local LLM endpoint (e.g., llama-server, vLLM, Ollama). Tested with Nail-Qwen3.6-35B.

### 1. Clone & Run
```bash
git clone https://github.com/mlalejandre/hierarchical-memory.git
cd hierarchical-memory
```

### 2. Launch the Web Interface
```bash
python3 iniciar_interfaz.py
```
*The native console will automatically launch at http://127.0.0.1:8765.*

---

## 🖥️ Web Console Overview

- **Real-Time Cognitive Pipeline:** Watch the agent switch between repair, decay, describe, and selftest in real time.
- **Chapter-Length Ingestion:** Paste entire textbook chapters or articles without context overflow; the built-in chunker extracts atomic concepts natively.
- **Tree Inspector:** Click 🌲 Ver Árbol to inspect your 8-level directory structure.
- **Epistemic Traceability:** View verified sources and citations behind every answer.

---

## 📐 Memory Decay Formula

Memories do not decay linearly. Vital facts (identity, safety directives) persist exponentially longer than transient details:

$$\text{Importancia Efectiva} = \text{Importancia} \cdot e^{-\frac{\text{Edad}}{\tau \cdot \text{Importancia}^3}}$$

When effective importance drops below the threshold, the fact is safely moved to _archivo/ (dormant), but is automatically restored if queried again in the future.

---

## 🛠️ Configuration

Configure your local model endpoint via environment variables:

```bash
export SERVER_IP="192.168.1.200"
export SERVER_USER="tu_usuario"
export LLM_BASE_URL="http://192.168.1.200:8087/v1"
export LLM_MODEL="nail-35b"
export MEM_TAU=20
```

---

## 📄 License
MIT License. Free for personal and commercial research.
