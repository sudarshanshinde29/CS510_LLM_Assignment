# Program Synthesis with LLMs – CS510 LLM_Assignment

This repository contains the implementation for a CS510: Advanced Information Retrieval assignment focused on program synthesis using Large Language Models (LLMs) such as Gemini-2.0-Flash and GPT-3.5-Turbo.

The goal of the assignment is to evaluate how effectively LLMs can generate correct, testable code in response to natural language programming problems. This simulates real-world developer scenarios where users describe functionality and expect auto-generated, compilable code.

## Setup & Execution

**Install requirements:**
```bash
pip install -r requirements.txt
```

```bash
pip uninstall google-generativeai
pip install --upgrade google-genai
```
**Add your API keys** to args while running below cmds as needed for Gemini or OpenAI.

**Run inference:**
```bash
python inference/run_gemini.py
python inference/run_gpt.py
```

**Run evaluation:**
```bash
python evaluator.py
```

## 📊 Results Summary

| Model           | Attempted | Pass@5 |
|------------------|-----------|--------|
| Gemini           | 46 / 56   | 26     |
| GPT-3.5 Turbo    | 56 / 56   | 6      |

**Note:** Gemini failed on 10 problems due to 503 errors.