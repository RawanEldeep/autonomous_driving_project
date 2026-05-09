Autonomous Driving with PPO — Submission
=========================================

Dependencies
------------
Python 3.10+

Install all dependencies with:

    pip install -r requirements.txt

Running the Notebook
--------------------
1. Create and activate a virtual environment:

    python -m venv .venv
    source .venv/bin/activate        # Linux / Mac
    .venv\Scripts\activate           # Windows

2. Install dependencies:

    pip install -r requirements.txt
    pip install ipykernel
    python -m ipykernel install --user --name=autonomous_driving --display-name "Python (autonomous_driving)"

3. Launch Jupyter and open the notebook:

    jupyter notebook notebooks/enhanced_ppo_pipeline.ipynb

4. Select kernel: Python (autonomous_driving)

5. Run all cells from top to bottom.

Folder Structure
----------------
notebooks/                  Main notebook
scripts/train_enhanced.py   Training pipeline and environment factory
src/models/                 PPO model and custom feature extractor
src/evaluation/             Metrics evaluator and plotting
src/envs/                   Custom environment wrappers
requirements.txt            Python dependencies
