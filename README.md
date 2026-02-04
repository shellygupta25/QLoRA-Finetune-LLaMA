# LLaMA-QLoRA-Pipeline

A reproducible, Dockerized pipeline for fine-tuning LLaMA models using QLoRA on domain-specific metadata classification tasks. This repository focuses on model adaptation infrastructure and training workflows. 

---

## Overview

This project implements an end-to-end pipeline for efficiently fine-tuning LLaMA-based large language models using Quantized Low-Rank Adaptation (QLoRA).

The primary goals of the repository are:
- to demonstrate memory-efficient LLM fine-tuning
- to provide a reproducible training setup

Although the example task is metadata classification, the pipeline is designed to be data-agnostic and adaptable to other domains.

---

## Key Features

- QLoRA-based fine-tuning
  - 4-bit quantization
  - Low-rank adapter injection
  - Reduced GPU memory footprint

- LLaMA model support
  - Compatible with LLaMA-family causal language models
  - Configurable base model checkpoints
  - I runa LLaMA 3.1 8B model

- Reproducible training environment
  - Fully Dockerized setup
  - Script-driven execution

- Modular design
  - Clear separation between data loading, model preparation, and training
  - Easy to extend or customize individual components

---

## Out of Scope

The following are intentionally excluded from this repository:
- Proprietary or internal datasets
- Domain-specific schemas or labels

---

## High-Level Pipeline

1. Data ingestion via a configurable interface
2. Base LLaMA model loading and quantization
3. QLoRA adapter initialization
4. Supervised fine-tuning for classification-style tasks
5. Output of trained adapter weights and training artifacts

---

## Getting Started

### Prerequisites

- Docker
- NVIDIA GPU with CUDA support

### Build the Docker Image

```bash

./model_training_on.sh

