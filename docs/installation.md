# Installation

## Requirements

- Python 3.10+
- No GPU required — all training runs on CPU

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Deactivate when done:

```bash
deactivate
```

## Verify the environment

```bash
python env/tag_env.py
```

Steps both environments with random actions and prints observations, rewards, and termination flags. Should complete without errors.
