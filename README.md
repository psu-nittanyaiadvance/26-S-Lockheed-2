# Project Title

Short description of the PyTorch-based ML project. Details will be added later.

**Setup**

This project uses a Python virtual environment and `requirements.txt` for dependencies.

**PowerShell Quickstart**

1. Create a virtual environment:

```powershell
# If python is on PATH
python -m venv .venv

# Or use the Windows launcher (this one worked for me) #antipythononPATHclub
py -m venv .venv
```

2. Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is blocked, allow local scripts for the current user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then re-run the activation command.

3. Install dependencies:

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

4. Deactivate when done:

```powershell
deactivate
```
