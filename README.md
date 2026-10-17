# Attention Interpretability Toolkit

[![tests](https://github.com/Rishichamp/attention-interpretability-toolkit/actions/workflows/tests.yml/badge.svg)](https://github.com/Rishichamp/attention-interpretability-toolkit/actions/workflows/tests.yml)

Tools for looking inside a transformer's attention and checking the maths behind positional encodings. It works with the GPT model from [transformer-from-scratch](https://github.com/Rishichamp/transformer-from-scratch), or with any model that returns `(batch, heads, T, T)` attention tensors.

## Features

**Attention analysis**

- **Entropy per head:** how focused or diffuse each head is.
- **Mean attention distance:** how far back each head looks.
- **Head similarity:** finds redundant heads.
- **Head-type classification:** diagonal, local or uniform.
- **Whole-model analysis** across all layers.

**Positional-encoding probes**

- **RoPE relative invariance:** verifies that the query·key score depends only on the distance between positions, not on the absolute positions.
- **ALiBi slopes:** measures each head's slope and checks it against the geometric formula.
- **Sinusoidal frequencies:** checks each dimension's frequency against theory.
- **Subject-verb agreement:** a behavioural probe of attention over growing distances.

## Tech stack

Python 3.11 · PyTorch · FastAPI · pytest

## Project structure

```
app/
  interpretability.py      entropy, distance, similarity, head classification
  probing.py               RoPE / ALiBi / sinusoidal / subject-verb probes
  model.py, tokenizer.py,  GPT model code from transformer-from-scratch,
  config.py,               included so this repo runs on its own
  positional_encoding.py
  main.py                  FastAPI server
tests/                     28 tests (analysis maths + API)
```

## Getting started

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python run.py                  # http://127.0.0.1:8001/docs
```

To analyse a **trained** model, copy `model.pt` and `tokenizer.json` from the Transformer Lab's `checkpoints/` folder into `./checkpoints/`, or set `M5P1_CHECKPOINT_DIR`. Without them, a small untrained model is used, which is fine for the probes.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/analyse/attention` | Entropy, distance and head types for an input |
| POST | `/probe/rope` | RoPE relative-position invariance |
| GET | `/probe/alibi` | ALiBi slope check |
| GET | `/probe/sinusoidal` | Sinusoidal frequency analysis |
| POST | `/probe/subject-verb` | Subject-verb agreement experiment |
| GET | `/health` | Health check, including which model is loaded |

## Tests

```bash
pytest tests/ -q
```

28 tests: the analysis maths (entropy bounds, distance, similarity, head classification, probe correctness) plus API tests that start the server and call every endpoint.

Built as part of Module 5 of a Generative AI engineering programme.
