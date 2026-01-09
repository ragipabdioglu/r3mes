# R3MES Installation Guide

This guide covers installation instructions for R3MES components across different operating systems.

## Table of Contents

1. [Choosing Your Interface](#choosing-your-interface)
2. [Prerequisites](#prerequisites)
3. [Desktop Launcher Installation](#desktop-launcher-installation) (Recommended - Easiest)
4. [Web Dashboard Installation](#web-dashboard-installation)
5. [CLI/Python SDK Installation](#cli-python-sdk-installation)
6. [Blockchain Node Installation](#blockchain-node-installation)
7. [Miner Engine Installation](#miner-engine-installation)
8. [Backend Installation](#backend-installation)
9. [Firewall Configuration](#firewall-configuration)

---

## Choosing Your Interface

Before installing, choose the interface that best fits your needs:

### 🖥️ Desktop Launcher (Recommended - Easiest Installation)

**Best for:** Node operators, miners, validators who want native desktop integration and local process management.

**Why Choose Desktop Launcher:**
- ✅ **Easiest Installation**: Download and install - setup wizard handles everything
- ✅ **Automatic Setup**: Hardware checks, IPFS setup, wallet creation all automated
- ✅ **Native Integration**: System tray, process management, background operation
- ✅ **No Technical Knowledge Required**: GUI-based setup wizard guides you through everything

**Installation**: See [Desktop Launcher Installation](#desktop-launcher-installation) below.

### 🌐 Web Dashboard (No Installation Required)

**Best for:** Users who want to use AI inference, stake tokens, participate in governance, or explore the network.

**Why Choose Web Dashboard:**
- ✅ **No Installation**: Access from any browser
- ✅ **Zero Setup**: Just open the URL and connect your wallet
- ✅ **Cross-Platform**: Works on any device/browser
- ✅ **Always Up-to-Date**: Automatic updates via web interface

**Installation**: See [Web Dashboard Installation](#web-dashboard-installation) below.

### 💻 CLI/Python SDK (For Developers)

**Best for:** Developers, advanced users, system administrators who need programmatic control.

**Why Choose CLI/SDK:**
- ✅ **Full Control**: Command-line interface for automation
- ✅ **Python SDK**: Programmatic access to R3MES APIs
- ✅ **Scriptable**: Easy to integrate into automated workflows

**Installation**: See [CLI/Python SDK Installation](#cli-python-sdk-installation) below.

**Quick Decision Guide:**
- **New User / Non-Technical**: Choose Desktop Launcher
- **Just Want to Use Features**: Choose Web Dashboard
- **Developer / Advanced User**: Choose CLI/SDK

---

---

## Prerequisites

### System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Disk Space**: 50GB+ free space
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended for mining)
- **Network**: Stable internet connection

### Required Software

- **Go**: 1.24.0 or higher
- **Python**: 3.10 or higher
- **Node.js**: 18.0 or higher
- **Docker**: Latest version (optional, for containerized deployment)
- **IPFS**: Embedded or standalone daemon

---

## Firewall Configuration

R3MES requires specific ports to be open for P2P connectivity and API access. Failure to configure firewall rules correctly can prevent miners from connecting to the network.

### Required Ports

| Port | Service | Protocol | Description |
|------|---------|----------|-------------|
| 26656 | Blockchain P2P | TCP | Tendermint P2P port for blockchain communication |
| 4001 | IPFS P2P | TCP | IPFS swarm port for decentralized storage |
| 26657 | Blockchain RPC | TCP | Tendermint RPC (local only, optional) |
| 1317 | Cosmos SDK REST | TCP | REST API (local only, optional) |
| 9090 | gRPC | TCP | gRPC queries (local only, optional) |
| 5001 | IPFS API | TCP | IPFS HTTP API (local only, optional) |

**Note**: Ports 26656 and 4001 are the only ports that need to be accessible from external networks. Other ports are only used locally.

### Windows Firewall Configuration

When you first start the R3MES miner or blockchain node on Windows, you may see a Windows Firewall prompt asking for permission to allow network access.

#### Automatic Prompt (Recommended)

1. When the miner starts, Windows Firewall will display a prompt asking if you want to allow the application through the firewall.
2. **Click "Allow Access"** to enable P2P connectivity.
3. Select both "Private networks" and "Public networks" checkboxes if prompted.
4. Click "OK" to confirm.

#### Manual Firewall Rule (Alternative)

If the automatic prompt doesn't appear or you need to configure rules manually:

1. Open **Windows Defender Firewall** from the Control Panel or Settings.
2. Click **"Advanced settings"** (requires administrator privileges).
3. In the left pane, click **"Inbound Rules"**.
4. Click **"New Rule"** in the right pane.
5. Select **"Port"** and click **Next**.
6. Select **TCP** and enter the port numbers:
   - For blockchain P2P: `26656`
   - For IPFS P2P: `4001`
7. Select **"Allow the connection"** and click **Next**.
8. Check all profile types (Domain, Private, Public) and click **Next**.
9. Enter a name (e.g., "R3MES Blockchain P2P") and click **Finish**.
10. Repeat steps 4-9 for each required port.

#### PowerShell Commands (Administrator)

You can also configure firewall rules using PowerShell:

```powershell
# Allow blockchain P2P port
New-NetFirewallRule -DisplayName "R3MES Blockchain P2P" -Direction Inbound -LocalPort 26656 -Protocol TCP -Action Allow

# Allow IPFS P2P port
New-NetFirewallRule -DisplayName "R3MES IPFS P2P" -Direction Inbound -LocalPort 4001 -Protocol TCP -Action Allow
```

### Linux Firewall Configuration (UFW)

If you're using UFW (Uncomplicated Firewall) on Linux:

```bash
# Allow blockchain P2P port
sudo ufw allow 26656/tcp comment 'R3MES Blockchain P2P'

# Allow IPFS P2P port
sudo ufw allow 4001/tcp comment 'R3MES IPFS P2P'

# Optional: Allow local-only ports (for development)
sudo ufw allow from 127.0.0.1 to any port 26657 comment 'R3MES Blockchain RPC (local)'
sudo ufw allow from 127.0.0.1 to any port 1317 comment 'R3MES REST API (local)'
sudo ufw allow from 127.0.0.1 to any port 9090 comment 'R3MES gRPC (local)'
sudo ufw allow from 127.0.0.1 to any port 5001 comment 'R3MES IPFS API (local)'

# Enable UFW (if not already enabled)
sudo ufw enable

# Check firewall status
sudo ufw status
```

### Linux Firewall Configuration (iptables)

If you're using iptables directly:

```bash
# Allow blockchain P2P port
sudo iptables -A INPUT -p tcp --dport 26656 -j ACCEPT

# Allow IPFS P2P port
sudo iptables -A INPUT -p tcp --dport 4001 -j ACCEPT

# Save iptables rules (Debian/Ubuntu)
sudo iptables-save > /etc/iptables/rules.v4

# Save iptables rules (RHEL/CentOS)
sudo service iptables save
```

### macOS Firewall Configuration

1. Open **System Preferences** → **Security & Privacy** → **Firewall**.
2. Click the lock icon and enter your password.
3. Click **"Firewall Options"**.
4. If the firewall is disabled, click **"Turn On Firewall"**.
5. For each R3MES application, click **"+"** and add:
   - R3MES miner/blockchain node executable
   - Allow incoming connections

Alternatively, use `pfctl` command-line tool:

```bash
# Add firewall rule (requires root)
sudo pfctl -f /etc/pf.conf

# Edit /etc/pf.conf to add:
pass in proto tcp from any to any port 26656
pass in proto tcp from any to any port 4001
```

### Testing Firewall Configuration

You can test if your ports are open using the R3MES setup wizard or manual tools:

#### Using R3MES Setup Wizard

The setup wizard automatically checks firewall ports during installation:

```bash
r3mes-miner setup
```

If ports are blocked, you'll see a warning with instructions.

#### Using netcat (Linux/macOS)

```bash
# Test if port is listening
nc -zv localhost 26656
nc -zv localhost 4001
```

#### Using Test-NetConnection (Windows PowerShell)

```powershell
# Test if port is listening
Test-NetConnection -ComputerName localhost -Port 26656
Test-NetConnection -ComputerName localhost -Port 4001
```

#### Using Online Port Checkers

For external connectivity testing, use online port checker tools:
- Visit https://www.yougetsignal.com/tools/open-ports/
- Enter your public IP address and port number (26656 or 4001)
- Click "Check" to verify external accessibility

**Note**: Online port checkers will only work if your router/firewall forwards these ports. For home networks, you may need to configure port forwarding on your router.

### Router Port Forwarding (Home Networks)

If you're running R3MES on a home network behind a router, you may need to configure port forwarding:

1. Access your router's admin panel (usually at `192.168.1.1` or `192.168.0.1`).
2. Navigate to **Port Forwarding** or **Virtual Server** settings.
3. Add forwarding rules:
   - **External Port**: 26656 → **Internal IP**: Your computer's IP → **Internal Port**: 26656
   - **External Port**: 4001 → **Internal IP**: Your computer's IP → **Internal Port**: 4001
4. Save and apply changes.

**Security Note**: Port forwarding exposes your services to the internet. Ensure your system is properly secured with strong passwords and up-to-date software.

### Common Firewall Issues

#### Issue: "Port already in use"

**Cause**: Another application is using the required port.

**Solution**:
```bash
# Find process using the port (Linux/macOS)
sudo lsof -i :26656
sudo lsof -i :4001

# Kill the process (replace PID with actual process ID)
kill -9 <PID>

# Or on Windows, use Task Manager to end the process
```

#### Issue: "Connection refused" from external peers

**Cause**: Firewall is blocking incoming connections.

**Solution**: Follow the firewall configuration steps above for your operating system.

#### Issue: Windows Firewall prompt doesn't appear

**Cause**: Windows Firewall may have existing rules or the prompt was dismissed.

**Solution**: Manually add firewall rules using the steps above or PowerShell commands.

---

## Desktop Launcher Installation

The Desktop Launcher is the **recommended and easiest way** to get started with R3MES. It provides a native desktop application with a setup wizard that handles all configuration automatically.

### Download and Install

**Windows:**
1. Download the Desktop Launcher installer (`.exe`) from the releases page
2. Run the installer and follow the setup wizard
3. The launcher will automatically check your system requirements

**macOS:**
1. Download the `.dmg` file from the releases page
2. Open the DMG and drag R3MES to Applications
3. Launch from Applications (you may need to allow it in Security & Privacy)

**Linux:**
1. Download the `.AppImage` or `.deb`/`.rpm` package from releases
2. For AppImage: `chmod +x R3MES-*.AppImage && ./R3MES-*.AppImage`
3. For .deb: `sudo dpkg -i r3mes-launcher_*.deb`
4. For .rpm: `sudo rpm -i r3mes-launcher_*.rpm`

### First Launch

When you first launch the Desktop Launcher, a setup wizard will guide you through:
1. Hardware compatibility check (GPU, RAM, disk)
2. Role selection (Miner, Serving, Validator, Proposer)
3. Blockchain configuration
4. Wallet setup (create new or import existing)
5. IPFS configuration (automatic)

**For detailed instructions, see [Desktop Launcher Documentation](10_desktop_launcher.md).**

---

## Web Dashboard Installation

The Web Dashboard requires **no installation** - just open it in your browser.

### Access the Web Dashboard

**Production:**
- URL: `https://dashboard.r3mes.network`
- Just open the URL in your browser
- No download or installation required

**Development/Local:**
If you want to run the Web Dashboard locally for development:

```bash
cd web-dashboard
npm install
npm run dev
```

Then open `http://localhost:${FRONTEND_PORT:-3000}` in your browser (or use the port configured via `FRONTEND_PORT` environment variable).

### First Use

1. Open the Web Dashboard in your browser
2. Click "Connect Wallet" in the top right
3. Select your wallet provider (Keplr, Leap, or Cosmostation)
4. Approve the connection request
5. Start using the features (Chat, Network Explorer, Staking, etc.)

**For detailed instructions, see [Web Dashboard Documentation](08_web_dashboard_command_center.md).**

---

## CLI/Python SDK Installation

For developers or advanced users who want command-line access.

### Installation

```bash
# Python 3.10+ required
python3 --version  # Should be 3.10+

# Install from PyPI
pip install r3mes

# Verify installation
r3mes-miner --version
```

### Windows (NVIDIA GPU)

On Windows, install CUDA-enabled PyTorch first:

```powershell
# Install CUDA-enabled PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install R3MES
pip install r3mes
```

### Setup

Run the setup wizard:

```bash
r3mes-miner setup
```

The wizard will guide you through:
1. System requirements check
2. Wallet creation/import
3. Blockchain configuration
4. IPFS setup (automatic)

**For detailed instructions, see [User Onboarding Guides](09_user_onboarding_guides.md).**

---

## Blockchain Node Installation

For users who want to run a standalone blockchain node (typically for validators):

See [Blockchain Infrastructure Documentation](01_blockchain_infrastructure.md) for detailed instructions.

---

## Miner Engine Installation

For users who want to install the miner engine separately (not using Desktop Launcher):

See [User Onboarding Guides - Miner Onboarding](09_user_onboarding_guides.md#miner-onboarding) for detailed instructions.

---

## Backend Installation

See the main documentation for backend installation instructions.

---

## Web Dashboard Installation

See the main documentation for web dashboard installation instructions.

---

## Desktop Launcher Installation

See the main documentation for desktop launcher installation instructions.

---

## Troubleshooting

For additional troubleshooting help, see [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).

