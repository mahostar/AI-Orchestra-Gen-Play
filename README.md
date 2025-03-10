# NotaGen on Windows Subsystem for Linux (WSL)

This guide provides instructions for setting up and running the NotaGen music generation model on Windows using WSL to resolve compatibility issues.

## Quick Reference

### First-Time Setup
```bash
# Install WSL (in PowerShell as Admin)
wsl --install

# After restart and Ubuntu setup, update system
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git wget curl python3-pip python3-dev python3-venv aria2

# Set up NotaGen
mkdir -p ~/projects/notagen
cd ~/projects/notagen
git clone https://github.com/deeplearn-art/NotaGen
cd NotaGen
python3 -m venv notagen_env
source notagen_env/bin/activate
pip install -r requirements.txt
pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install accelerate optimum gradio aria2p

# Download model and run
cd gradio
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ElectricAlexis/NotaGen/resolve/main/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth -o weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth
python demo.py
```

### Reusing Your WSL Environment
```bash
# Open WSL (from Command Prompt or PowerShell)
wsl -d Ubuntu
# Navigate to project and activate environment
cd ~/projects/notagen/NotaGen
source notagen_env/bin/activate

# Run the demo
cd gradio
python demo.py
```

### Managing WSL
```bash
# List distributions
wsl --list --verbose

# Shutdown WSL
wsl --shutdown

# Remove a distribution (WARNING: Deletes all data)
wsl --unregister Ubuntu
```

## Detailed Guide

### 1. Installing WSL

1. **Open PowerShell as Administrator**
   - Right-click on the Start button
   - Select "Windows PowerShell (Admin)" or "Windows Terminal (Admin)"

2. **Install WSL with Ubuntu**
   ```powershell
   wsl --install
   ```

3. **Restart your computer**

4. **Complete Ubuntu Setup**
   - Create a username and password when prompted

### 2. Setting Up the Linux Environment

1. **Update and Upgrade Ubuntu**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. **Install Essential Tools**
   ```bash
   sudo apt install -y build-essential git wget curl
   sudo apt install -y python3-pip python3-dev python3-venv
   sudo apt install -y aria2
   ```

### 3. Setting Up NotaGen

1. **Create Project Directory and Clone Repository**
   ```bash
   mkdir -p ~/projects/notagen
   cd ~/projects/notagen
   git clone https://github.com/deeplearn-art/NotaGen
   cd NotaGen
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   python3 -m venv notagen_env
   source notagen_env/bin/activate
   ```

3. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install accelerate optimum gradio aria2p
   ```

### 4. Running NotaGen

1. **Download Model Weights**
   ```bash
   cd ~/projects/notagen/NotaGen/gradio
   aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ElectricAlexis/NotaGen/resolve/main/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth -o weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth
   ```

2. **Run the Demo**
   ```bash
   python demo.py
   ```
   Access the web interface at http://127.0.0.1:7860

### 5. Reusing Your WSL Environment

When you want to use NotaGen again later:

1. **Open WSL**
   ```
   wsl
   ```
   
2. **Go to Project and Activate Environment**
   ```bash
   cd ~/projects/notagen/NotaGen
   source notagen_env/bin/activate
   ```

3. **Run the Demo**
   ```bash
   cd gradio
   python demo.py
   ```

### 6. Accessing Files

- **Windows files in WSL**: Available at `/mnt/c/`, `/mnt/d/`, etc.
- **WSL files in Windows**: Access via `\\wsl$\Ubuntu\home\yourusername\projects\notagen`

### 7. Troubleshooting

If model file is corrupted:
```bash
rm ~/projects/notagen/NotaGen/gradio/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ElectricAlexis/NotaGen/resolve/main/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth -o weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth
```

If WSL doesn't start:
```powershell
wsl --status
wsl --update
wsl --shutdown
```

## Accessing Windows Files from WSL

Your Windows drives are mounted in WSL under the `/mnt/` directory:

- C: drive is at `/mnt/c/`
- D: drive is at `/mnt/d/`
- etc.

To access your Windows files:

```bash
cd /mnt/c/Users/YourUsername/Documents
```

## Accessing WSL Files from Windows

You can access your WSL files from Windows Explorer by:

1. Opening Windows Explorer
2. Typing `\\wsl$\Ubuntu` in the address bar
3. Navigating to your files (e.g., `\\wsl$\Ubuntu\home\yourusername\projects\notagen`)

---

This guide should help you set up and manage NotaGen on WSL. If you encounter any issues not covered here, please refer to the official documentation or seek help from the NotaGen community. 