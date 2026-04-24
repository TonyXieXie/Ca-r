# CarRacing PPO

Python scripts for training and running a pixel-based PPO policy on Gymnasium `CarRacing-v3`.

## Environment

Recommended: Python 3.12+ with a project-local virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you need CUDA acceleration, install the PyTorch build that matches your CUDA driver from the official PyTorch instructions, then install the remaining packages from `requirements.txt`.

## Common Commands

Train a new policy:

```powershell
.\run_train_carracing.ps1
```

Run an existing checkpoint:

```powershell
.\run_best_carracing.ps1 -RunDir runs\your_run_folder
```

Play manually:

```powershell
.\start_carracing.ps1
```

## Git Notes

The repository should include source code, scripts, documentation, and dependency metadata. Local virtual environments, training runs, logs, checkpoints, videos, screenshots, and other generated artifacts are ignored by `.gitignore`.

