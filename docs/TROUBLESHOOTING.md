# R3MES Troubleshooting Guide

This guide helps you resolve common issues encountered when using R3MES.

## Table of Contents

1. [Firewall and Network Issues](#firewall-and-network-issues)
2. [Blockchain Node Issues](#blockchain-node-issues)
3. [Miner Engine Issues](#miner-engine-issues)
4. [IPFS Issues](#ipfs-issues)
5. [Backend Issues](#backend-issues)
6. [Web Dashboard Issues](#web-dashboard-issues)
7. [Desktop Launcher Issues](#desktop-launcher-issues)
8. [General Issues](#general-issues)

---

## Firewall and Network Issues

### Port Already in Use

**Symptom**: Error message indicating that a port (26656, 4001, etc.) is already in use.

**Cause**: Another process is using the required port, or a previous R3MES process didn't shut down properly.

**Solution**:

#### Linux/macOS:
```bash
# Find the process using the port
sudo lsof -i :26656
sudo lsof -i :4001

# Kill the process (replace <PID> with the actual process ID)
kill -9 <PID>

# Or kill all processes using the port
sudo fuser -k 26656/tcp
sudo fuser -k 4001/tcp
```

#### Windows:
```powershell
# Find the process using the port
netstat -ano | findstr :26656
netstat -ano | findstr :4001

# Kill the process (replace <PID> with the actual process ID)
taskkill /PID <PID> /F
```

**Prevention**: Always use the graceful shutdown commands:
- `r3mes-miner stop` (for miner)
- Desktop launcher: Click "Stop" buttons before closing the application

### Firewall Blocking Connections

**Symptom**: Miners cannot connect to the network, or external peers cannot reach your node.

**Cause**: Firewall rules are blocking incoming connections on required ports (26656, 4001).

**Solution**: See the [Firewall Configuration](./INSTALLATION.md#firewall-configuration) section in the Installation Guide.

**Quick Check**:
```bash
# Linux/macOS: Check if ports are listening
sudo netstat -tuln | grep -E '26656|4001'

# Windows: Check if ports are listening
netstat -an | findstr -E '26656|4001'
```

### Connection Timeout / Network Unreachable

**Symptom**: Timeout errors when trying to connect to blockchain or IPFS nodes.

**Cause**: Network connectivity issues, incorrect URLs, or firewall blocking outbound connections.

**Solution**:
1. Verify internet connectivity: `ping google.com`
2. Check if the target host is reachable: `ping <hostname>`
3. Verify URLs in configuration files:
   - Blockchain gRPC URL: Should be `localhost:9090` (local) or `node.r3mes.network:9090` (mainnet)
   - IPFS URL: Should be `http://localhost:5001` (local)
4. Check firewall rules for outbound connections (usually not blocked by default)

---

## Blockchain Node Issues

### Node Not Syncing

**Symptom**: Block height is not increasing, or sync percentage stuck.

**Cause**: Network connectivity issues, insufficient peers, or database corruption.

**Solution**:
1. Check node logs: `journalctl -u remesd -f` (systemd) or check log files
2. Verify peer connections:
   ```bash
   curl http://localhost:26657/net_info
   ```
3. Restart the node:
   ```bash
   sudo systemctl restart remesd
   ```
4. If persistent, try resetting the node (backup data first):
   ```bash
   remesd unsafe-reset-all
   ```

### Transaction Failures

**Symptom**: Transactions fail with "insufficient funds" or "invalid sequence" errors.

**Cause**: Insufficient balance, incorrect sequence number, or network fee issues.

**Solution**:
1. Check wallet balance:
   ```bash
   remesd query bank balances <your-address>
   ```
2. Request tokens from faucet (if on testnet):
   - Use the backend faucet endpoint: `POST /faucet/claim`
   - Or use the miner setup wizard which automatically requests faucet tokens
3. Verify transaction sequence:
   ```bash
   remesd query account <your-address>
   ```

---

## Miner Engine Issues

### GPU Not Detected

**Symptom**: Miner falls back to CPU mode or shows "CUDA not available".

**Cause**: CUDA not installed, incorrect CUDA version, or GPU drivers outdated.

**Solution**:
1. Check CUDA installation:
   ```bash
   nvidia-smi
   ```
2. Verify PyTorch CUDA support:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.version.cuda)
   ```
3. Install/upgrade CUDA-enabled PyTorch:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
4. Update GPU drivers (Windows: NVIDIA GeForce Experience, Linux: `sudo apt install nvidia-driver-xxx`)

### Out of Memory (OOM) Errors

**Symptom**: "CUDA out of memory" or "RuntimeError: CUDA out of memory".

**Cause**: GPU doesn't have enough VRAM for the model, or multiple processes are using GPU.

**Solution**:
1. Reduce model size in configuration (if supported)
2. Close other GPU-intensive applications
3. Use gradient accumulation to reduce memory usage:
   ```yaml
   gradient_accumulation_steps: 4  # Increase this value
   ```
4. Limit GPU memory usage:
   ```bash
   export CUDA_VISIBLE_DEVICES=0  # Use only first GPU
   ```

### Miner Not Submitting Gradients

**Symptom**: Miner trains successfully but doesn't submit gradients to the blockchain.

**Cause**: Wallet issues, insufficient gas fees, or blockchain connection problems.

**Solution**:
1. Check wallet balance (ensure sufficient funds for gas fees)
2. Verify blockchain connection:
   ```bash
   r3mes-miner status
   ```
3. Check miner logs for error messages
4. Verify wallet is properly configured in `~/.r3mes/config/config.json`

### Time Synchronization Errors

**Symptom**: "Invalid timestamp" errors or transactions rejected due to time drift.

**Cause**: System clock is out of sync with network time.

**Solution**:
1. Sync system time (Linux):
   ```bash
   sudo ntpdate -s time.nist.gov
   # Or use systemd-timesyncd
   sudo timedatectl set-ntp true
   ```
2. Sync system time (Windows):
   - Right-click on clock → "Adjust date/time"
   - Enable "Set time automatically"
3. The miner setup wizard now includes automatic time sync checks

---

## IPFS Issues

### IPFS Not Starting

**Symptom**: IPFS daemon fails to start or crashes immediately.

**Cause**: Port conflicts, permission issues, or corrupted IPFS repository.

**Solution**:
1. Check if IPFS port (5001, 4001) is already in use (see "Port Already in Use" above)
2. Reset IPFS repository (backup data first):
   ```bash
   ipfs repo gc  # Garbage collect first
   # If still fails:
   rm -rf ~/.ipfs
   ipfs init
   ```
3. Check IPFS logs for specific error messages

### IPFS Content Not Accessible

**Symptom**: IPFS content cannot be retrieved or pinned files are not accessible via public gateway.

**Cause**: IPFS node not connected to network, port blocking, or content not properly propagated.

**Solution**:
1. Check IPFS swarm peers:
   ```bash
   ipfs swarm peers
   ```
2. Verify IPFS is listening on correct ports:
   ```bash
   ipfs config Addresses.Swarm
   ```
3. Test public gateway access:
   ```bash
   curl https://ipfs.io/ipfs/<YOUR_CID>
   ```
4. Ensure firewall allows port 4001 (see [Firewall Configuration](./INSTALLATION.md#firewall-configuration))

---

## Backend Issues

### API Endpoints Not Responding

**Symptom**: HTTP 500 errors or connection refused when accessing backend API.

**Cause**: Backend service not running, incorrect configuration, or database issues.

**Solution**:
1. Check if backend is running:
   ```bash
   ps aux | grep "r3mes-backend"
   # Or check systemd service
   sudo systemctl status r3mes-backend
   ```
2. Check backend logs for errors
3. Verify environment variables are set correctly (see `.env.production.example`)
4. Test database connection:
   ```bash
   sqlite3 backend/database.db "SELECT 1;"
   ```

### Model Loading Failures

**Symptom**: Backend fails to load AI models or inference returns errors.

**Cause**: Model files missing, incorrect model path, or insufficient memory.

**Solution**:
1. Verify model path in configuration:
   ```bash
   echo $BASE_MODEL_PATH
   ls -la $BASE_MODEL_PATH
   ```
2. Check available disk space (models can be several GB)
3. Verify model file integrity (re-download if corrupted)

---

## Web Dashboard Issues

### Dashboard Not Loading

**Symptom**: Blank page or "Connection refused" errors in browser.

**Cause**: Web server not running, incorrect API URLs, or CORS issues.

**Solution**:
1. Check if Next.js server is running:
   ```bash
   ps aux | grep "next"
   # Or check systemd service
   sudo systemctl status r3mes-dashboard
   ```
2. Verify environment variables (especially `NEXT_PUBLIC_BACKEND_URL`)
3. Check browser console for CORS errors
4. Verify backend is accessible from the frontend URL

### Wallet Not Connecting

**Symptom**: Keplr or other wallet extensions fail to connect.

**Cause**: Incorrect chain configuration, network mismatch, or wallet extension issues.

**Solution**:
1. Verify chain ID matches (should be `remes-mainnet-1` for mainnet)
2. Check RPC endpoint is correct and accessible
3. Try disconnecting and reconnecting the wallet
4. Clear browser cache and reload the page

---

## Desktop Launcher Issues

### Launcher Crashes on Startup

**Symptom**: Desktop launcher closes immediately after opening.

**Cause**: Missing dependencies, corrupted installation, or permission issues.

**Solution**:
1. Check launcher logs (usually in `~/.config/r3mes/` or `%APPDATA%\R3MES\`)
2. Reinstall the launcher
3. Verify all dependencies are installed (Rust toolchain, Node.js, etc.)
4. Run from terminal to see error messages:
   ```bash
   ./r3mes-launcher
   ```

### Processes Not Starting

**Symptom**: Clicking "Start" buttons in launcher doesn't start processes.

**Cause**: Executables not found, permission issues, or configuration errors.

**Solution**:
1. Check if Python miner executable exists and is executable
2. Verify IPFS daemon path is correct
3. Check launcher logs for specific error messages
4. Ensure all required dependencies are installed

### Zombie Processes

**Symptom**: Processes continue running after closing the launcher.

**Cause**: Launcher didn't properly clean up child processes on exit.

**Solution**:
1. The launcher now includes automatic cleanup on exit (graceful shutdown)
2. If processes persist, manually kill them (see "Port Already in Use" section)
3. Restart the launcher to ensure cleanup runs

---

## General Issues

### Python Dependency Conflicts

**Symptom**: Import errors or version conflicts when running miner.

**Cause**: Conflicting Python package versions or virtual environment issues.

**Solution**:
1. Use a fresh virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate  # Windows
   ```
2. Reinstall dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```
3. For Windows EXE builds, use PyInstaller with proper spec file (see production deployment plan)

### Version Mismatch Warnings

**Symptom**: Warning about outdated miner or backend version.

**Cause**: Miner version is older than the minimum required version.

**Solution**:
1. Check current version:
   ```bash
   r3mes-miner version
   ```
2. Update to latest version:
   ```bash
   pip install --upgrade r3mes-miner
   ```
3. The miner now includes automatic version checks at startup (warnings only, won't block)

### Slow Performance

**Symptom**: Mining or training is slower than expected.

**Cause**: CPU mode instead of GPU, insufficient resources, or network bottlenecks.

**Solution**:
1. Verify GPU is being used (check `nvidia-smi` output)
2. Monitor system resources (CPU, RAM, disk I/O)
3. Check network bandwidth for IPFS downloads
4. Reduce model size or training batch size if resource-constrained

---

## Getting Help

If you're still experiencing issues after trying the solutions above:

1. Check the [main documentation](./README.md) for detailed guides
2. Review logs for specific error messages
3. Search GitHub issues for similar problems
4. Create a new issue with:
   - Your operating system and version
   - R3MES component versions
   - Error messages and logs
   - Steps to reproduce the issue

---

## Related Documentation

- [Installation Guide](./INSTALLATION.md)
- [Production Deployment](./12_production_deployment.md)
- [Environment Variables](./16_environment_variables.md)

