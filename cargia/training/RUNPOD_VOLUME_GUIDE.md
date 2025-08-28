# RunPod Volume Mounting Guide

This guide explains how to work with mounted volumes when SSH-ing into RunPod Docker containers.

## 🔗 **Volume Mounting in RunPod**

### **1. Pre-Mounted Volumes (Recommended)**
When you create a RunPod pod, specify volumes that get mounted **before** the container starts:

```yaml
# In your RunPod pod configuration
volumes:
  - name: data-volume
    mountPath: /data
    size: 100Gi
  - name: model-weights
    mountPath: /weights
    size: 50Gi
  - name: outputs
    mountPath: /outputs
    size: 20Gi
```

These volumes are automatically available inside the container at the specified mount paths.

### **2. Runtime Volume Mounting (Advanced)**
You can also mount volumes after the container is running, but this requires additional setup.

## 📁 **Accessing Mounted Volumes Inside Container**

### **Step 1: SSH into the Container**
```bash
# From your local machine
ssh root@<container-ip> -p <ssh-port>
```

### **Step 2: Check What's Already Mounted**
Once inside the container, check what volumes are available:

```bash
# List all mounted filesystems
df -h

# Check mount points
mount | grep -E "(data|weights|outputs)"

# List contents of common mount directories
ls -la /data
ls -la /weights
ls -la /outputs
ls -la /workspace
```

### **Step 3: Navigate and Use Mounted Volumes**
```bash
# Navigate to your data directory
cd /data

# List your datasets
ls -la

# Check model weights
ls -la /weights

# Set up output directory
mkdir -p /outputs/training_runs
```

## 🗂️ **Common RunPod Mount Points**

RunPod typically mounts volumes at these locations:

| Mount Point | Purpose | Typical Contents |
|-------------|---------|------------------|
| `/data` | Dataset storage | Training data, source files |
| `/weights` | Model checkpoints | Pre-trained models, LoRA weights |
| `/outputs` | Training outputs | Logs, checkpoints, results |
| `/workspace` | Code/project files | Your training scripts, configs |

## 🚀 **Using Your Updated Configs**

Your YAML configs now use RunPod-compatible paths:

### **Local Development**
```bash
python train_cli.py --config configs/step_1_overfit_single.yaml --local
```

### **RunPod Cloud Deployment**
```bash
python train_cli.py --config configs/step_1_overfit_single.yaml --cloud
```

## 🔧 **Docker Image Setup**

### **Important: Package Installation**
The Dockerfile now includes `pip install -e .` to install the `cargia` package. This is required for the imports to work correctly.

### **Rebuild Your Docker Image**
After updating the Dockerfile, you need to rebuild your image:

```bash
# From your local project directory
docker build -t cargia-training .

# Or if using docker-compose
docker-compose build
```

### **Verify Package Installation**
Once inside the container, test that the package is installed:

```bash
# Test imports
python training/test_imports.py

# Or test manually
python -c "from cargia.training.trainer import CargiaGoogleGemma3Trainer; print('Import successful!')"
```

The `--cloud` flag automatically uses these paths:
- **Data**: `/data/solves_and_thoughts`
- **Source**: `/data/arc_agi_2_reformatted`
- **Model**: `/weights/gemma3-4b-it-ORIGINAL`
- **Outputs**: `/outputs`

## 🔍 **Troubleshooting Volume Access**

### **Problem: Can't see mounted volumes**
```bash
# Check if volumes are mounted
mount | grep -E "(data|weights|outputs)"

# Check disk usage
df -h

# Look for volume devices
lsblk
```

### **Problem: Permission denied**
```bash
# Check ownership
ls -la /data

# Fix permissions if needed
chmod -R 755 /data
chown -R root:root /data
```

### **Problem: Volume not showing expected content**
```bash
# Check volume mount status
docker volume ls

# Inspect volume details
docker volume inspect <volume-name>
```

## 📋 **Complete RunPod Workflow**

### **1. Create Pod with Volumes**
```yaml
# In RunPod pod configuration
volumes:
  - name: training-data
    mountPath: /data
    size: 100Gi
  - name: model-weights
    mountPath: /weights
    size: 50Gi
  - name: training-outputs
    mountPath: /outputs
    size: 20Gi
```

### **2. SSH into Container**
```bash
ssh root@<container-ip> -p <ssh-port>
```

### **3. Verify Volumes**
```bash
# Check mounts
df -h
ls -la /data /weights /outputs

# Verify your data is there
ls -la /data/solves_and_thoughts
ls -la /weights/gemma3-4b-it-ORIGINAL
```

### **4. Run Training**
```bash
# Navigate to your project
cd /workspace/cargia

# Run training with cloud config
python train_cli.py --config configs/step_1_overfit_single.yaml --cloud
```

### **5. Check Outputs**
```bash
# Monitor training progress
tail -f /outputs/training_runs/*/training.log

# Check saved checkpoints
ls -la /outputs/training_runs/*/checkpoint-*
```

## 💡 **Best Practices**

1. **Always verify volumes before training**: Check that `/data`, `/weights`, and `/outputs` contain expected content
2. **Use absolute paths**: Your configs now use absolute paths like `/data/...` for cloud deployment
3. **Monitor disk space**: Use `df -h` to ensure you have enough space for outputs
4. **Backup important outputs**: Copy results from `/outputs` to persistent storage if needed
5. **Test locally first**: Use `--local` flag to test configs before deploying to RunPod

## 🆘 **Getting Help**

If you encounter volume mounting issues:
1. Check the RunPod pod configuration for volume definitions
2. Verify volumes are properly attached in the RunPod dashboard
3. Use `docker volume ls` and `docker volume inspect` inside the container
4. Check RunPod documentation for volume troubleshooting 