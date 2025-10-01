# Artificial Intelligence for Engineers

![Background](/references/AIEng_Background.png)

This repository contains course materials for AI for engineers course (W10IME-SD0033S).  

This 30-hour "Artificial Intelligence for Engineers" course introduces doctoral students to AI fundamentals and applications in engineering. It combines theoretical lectures on machine learning, deep learning, and generative AI with hands-on Python coding sessions. Students develop practical skills through laboratory exercises and apply their knowledge in seminar presentations related to their PhD research. The course aims to equip engineers to leverage AI to solve complex problems.

## 📥 Cloning the Repository 

Clone the repository to your local machine:

```bash
git clone https://github.com/w10k57-education/ai_engineering.git
```

## 🛠️ Environment Setup 

This project uses `uv` for dependency management. Make sure you have `uv` installed on your system.

To set up the project environment, run:

```bash
uv sync
```

This command will create a virtual environment and install all dependencies specified in the `pyproject.toml` file.
Now, you can run commands directly in the environment without activating it:

```bash
uv run python your_script.py
```