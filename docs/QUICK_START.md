# R3MES Quick Start Guide

Get up and running with R3MES in under 5 minutes. This guide covers the essential steps to start mining, running a node, or using the R3MES network.

## Overview

**R3MES** (Revolutionary Resource-Efficient Machine Learning Ecosystem) is a blockchain protocol that combines Proof of Useful Work (PoUW) consensus with AI model training. Unlike traditional blockchain mining, R3MES enables miners to earn tokens by training AI models, creating a self-sustaining ecosystem for decentralized machine learning.

## Choosing Your Interface

R3MES provides multiple interfaces depending on your needs:

### 🖥️ Desktop Launcher (Recommended for Node Operators & Miners)

**Best for:**
- Node operators running blockchain nodes, miners, or validators
- Users who need local process management
- System administrators who want native desktop integration
- Users requiring embedded IPFS management

**Key Features:**
- Native desktop application (Tauri-based, cross-platform)
- Setup wizard with hardware compatibility checks
- Process management (Start/Stop Node, Miner, IPFS)
- System status panel with real-time monitoring
- Live logs viewer with filtering and search
- Wallet management with local keystore
- Model downloader with progress tracking
- System tray integration for background operation

**Platforms:** Windows, macOS, Linux

### 🌐 Web Dashboard (Recommended for Users & Investors)

**Best for:**
- End users who want to use AI inference (chat interface)
- Investors and network observers
- Users who want to stake tokens or participate in governance
- Developers exploring the network through a web interface

**Key Features:**
- Unified web interface (Next.js 14)
- AI chat interface with streaming responses
- Mining dashboard with statistics and earnings
- Network explorer with 3D globe visualization
- Wallet management and transaction history
- Staking and governance participation
- Real-time network statistics

**Access:** Open in any modern web browser

### 💻 CLI/Python SDK (Recommended for Developers)

**Best for:**
- Developers integrating R3MES into applications
- Advanced users who prefer command-line interfaces
- Automated scripts and bots
- System administrators managing multiple nodes

**Key Features:**
- Python package (`pip install r3mes`)
- Command-line tools for mining, serving, and validation
- Python SDK for programmatic access
- Full control over configuration and parameters

---

## Quick Start Options

### Option 1: Desktop Launcher (Easiest Path for Node Operators)

The Desktop Launcher is the recommended way to get started if you want to run a node, mine, or operate infrastructure components.

#### Step 1: Download and Install

**Windows:**
1. Download the Desktop Launcher installer from the releases page
2. Run the installer and follow the setup wizard
3. The launcher will automatically check your system requirements

**macOS:**
1. Download the `.dmg` file from the releases page
2. Open the DMG and drag R3MES to Applications
3. Launch from Applications (you may need to allow it in Security & Privacy)

**Linux:**
1. Download the `.AppImage` or `.deb`/.`rpm` package from releases
2. For AppImage: `chmod +x R3MES-*.AppImage && ./R3MES-*.AppImage`
3. For .deb: `sudo dpkg -i r3mes-launcher_*.deb`
4. For .rpm: `sudo rpm -i r3mes-launcher_*.rpm`

#### Step 2: First Launch - Setup Wizard

When you first launch the Desktop Launcher, a setup wizard will guide you through:

1. **Hardware Compatibility Check**
   - GPU detection (NVIDIA recommended)
   - Disk space verification (100GB+ recommended)
   - RAM check (16GB+ recommended)
   - CUDA version check

2. **Role Selection**
   - Choose which roles your node will perform:
     - **Miner**: Train AI models and earn tokens
     - **Serving Node**: Provide AI inference services
     - **Validator**: Participate in consensus
     - **Proposer**: Aggregate gradients from miners

3. **Blockchain Configuration**
   - Network selection (Mainnet/Testnet)
   - RPC endpoint configuration
   - Chain ID verification

4. **Wallet Setup**
   - Create new wallet or import existing
   - Wallet encryption and backup
   - Initial balance check

5. **IPFS Configuration**
   - Automatic embedded IPFS daemon setup
   - Port configuration (default: 5001)
   - Storage path selection

#### Step 3: Start Mining/Operating

After setup is complete:

1. Click **"Start Node"** to begin syncing with the blockchain
2. Click **"Start Miner"** (if you selected Miner role) to begin training
3. Monitor status in the System Status Panel:
   - Chain sync progress
   - IPFS peer connections
   - Model download status
   - Node health

#### Step 4: Register Your Node Roles

After local setup, register your selected roles on the blockchain:

1. Open the Web Dashboard (link provided in launcher)
2. Navigate to the **Role Management** page (`/roles`)
3. Connect your wallet (Keplr or compatible wallet)
4. Select your roles and set stake amount
5. Submit the registration transaction

**Note:** For Validator and Proposer roles, authorization may be required. See the [Role Registration Guide](role_registration_guide.md) for details.

#### Next Steps

- View detailed guide: [Desktop Launcher Documentation](10_desktop_launcher.md)
- Troubleshooting: Check the logs viewer in the launcher
- Community support: Join our Discord or GitHub Discussions

---

### Option 2: Web Dashboard (Easiest Path for Users)

The Web Dashboard is the easiest way to start using R3MES if you want to use AI inference, stake tokens, or explore the network.

#### Step 1: Access the Web Dashboard

The Web Dashboard is typically available at:
- **Production**: `https://dashboard.r3mes.network` (or your configured domain via `DOMAIN` environment variable)
- **Development**: `http://localhost:${FRONTEND_PORT:-3000}` (if running locally, port configurable via `FRONTEND_PORT` environment variable)

No installation required - just open it in your browser!

#### Step 2: Connect Your Wallet

1. Click the **"Connect Wallet"** button in the top right
2. Select your wallet provider:
   - **Keplr** (Recommended)
   - **Leap Wallet**
   - **Cosmostation**
3. Approve the connection request
4. Your wallet address will be displayed

#### Step 3: Get Started Tokens (Faucet)

If you're new to the network:

1. Navigate to the **Faucet** page (`/faucet`)
2. Enter your wallet address
3. Click **"Claim Tokens"** to receive test tokens
4. Wait for transaction confirmation

**Note:** Rate limit applies (typically once per day per address).

#### Step 4: Explore Features

The Web Dashboard provides several key pages:

- **Chat** (`/chat`): AI inference interface with streaming responses
- **Mine** (`/mine`): Mining statistics and earnings (if you're mining)
- **Network** (`/network`): 3D globe visualization, miners table, recent blocks
- **Wallet** (`/wallet`): Transaction history, API keys, balance management
- **Staking** (`/staking`): Delegate tokens, claim rewards
- **Roles** (`/roles`): Register node roles on the blockchain
- **Settings** (`/settings`): Configuration management

#### Next Steps

- View detailed guide: [Web Dashboard Documentation](08_web_dashboard_command_center.md)
- API Reference: [API Documentation](13_api_reference.md)
- Troubleshooting: Check the Help page (`/help`)

---

### Option 3: CLI/Python SDK (Developer Path)

For developers or advanced users who want full programmatic control.

#### Step 1: Install Python Package

```bash
# Ensure Python 3.10+ is installed
python3 --version

# Install R3MES package
pip install r3mes

# Verify installation
r3mes-miner --version
```

**Windows Note:** If you have an NVIDIA GPU, install CUDA-enabled PyTorch first:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install r3mes
```

#### Step 2: Run Setup Wizard

```bash
r3mes-miner setup
```

The setup wizard will:
- Create or import a wallet
- Configure blockchain connection
- Set up IPFS (embedded daemon)
- Configure mining parameters

#### Step 3: Start Mining

```bash
# Start mining
r3mes-miner start

# Or with custom parameters
r3mes-miner start --batch-size 5 --pause-seconds 2

# Continuous mining mode
r3mes-miner start --continuous
```

#### Step 4: Use Python SDK

```python
from r3mes import R3MESClient

# Initialize client
client = R3MESClient(
    backend_url="https://api.r3mes.network",
    wallet_address="remes1..."
)

# Send chat message
async for chunk in client.chat("What is R3MES?"):
    print(chunk, end='')

# Get network stats
stats = await client.get_network_stats()
print(f"Active miners: {stats['active_miners']}")
```

#### Next Steps

- View detailed guide: [User Onboarding Guides](09_user_onboarding_guides.md)
- API Reference: [Python SDK Documentation](13_api_reference.md)
- Examples: Check the `examples/` directory in the repository

---

## Interface Comparison

| Feature | Desktop Launcher | Web Dashboard | CLI/SDK |
|---------|-----------------|---------------|---------|
| **Node Management** | ✅ Native | ❌ | ✅ Full Control |
| **Mining** | ✅ GUI | ❌ | ✅ Full Control |
| **IPFS Management** | ✅ Embedded | ❌ | ✅ Full Control |
| **AI Chat** | ❌ | ✅ Streaming | ✅ Programmatic |
| **Network Explorer** | ❌ | ✅ 3D Globe | ✅ API Access |
| **Staking** | ❌ | ✅ Full UI | ✅ CLI/API |
| **Governance** | ❌ | ✅ Voting UI | ✅ CLI/API |
| **Wallet Signing** | ✅ Local | ✅ Browser Wallet | ✅ Programmatic |
| **System Tray** | ✅ Native | ❌ | N/A |
| **Process Monitoring** | ✅ Real-time | ❌ | ✅ Logs |
| **Cross-Platform** | ✅ Windows/macOS/Linux | ✅ Any Browser | ✅ Any OS |

## Troubleshooting

### Common Issues

#### Desktop Launcher

**Issue:** Setup wizard fails hardware check
- **Solution:** Ensure you meet minimum requirements (16GB RAM, 100GB disk, NVIDIA GPU for mining)

**Issue:** IPFS daemon won't start
- **Solution:** Check if port 5001 is already in use, or configure a different port

**Issue:** Node won't sync
- **Solution:** Check firewall rules (port 26656 must be open), verify RPC endpoint is accessible

#### Web Dashboard

**Issue:** Wallet connection fails
- **Solution:** Ensure Keplr/Leap is installed and unlocked, check if R3MES network is added

**Issue:** Chat responses don't stream
- **Solution:** Check browser console for errors, verify backend API is accessible

#### CLI/SDK

**Issue:** `pip install r3mes` fails
- **Solution:** Ensure Python 3.10+ is installed, update pip: `pip install --upgrade pip`

**Issue:** Miner won't start
- **Solution:** Check GPU drivers (NVIDIA), verify CUDA is installed, check logs in `~/.r3mes/logs/`

### Getting Help

- **Documentation**: See the [full documentation index](README.md)
- **GitHub Issues**: Report bugs or request features
- **Discord Community**: Get help from the community
- **Help Page**: Access the Help page in the Web Dashboard (`/help`)

## Next Steps

Now that you're set up, explore more:

1. **Learn More About R3MES**
   - [Project Overview](00_project_summary.md)
   - [Architecture Documentation](ARCHITECTURE_OVERVIEW.md)
   - [Economic Model](04_economic_incentives.md)

2. **Deepen Your Understanding**
   - [Blockchain Infrastructure](01_blockchain_infrastructure.md)
   - [AI Training System](02_ai_training_system.md)
   - [Security & Verification](03_security_verification.md)

3. **Advanced Topics**
   - [Production Deployment](12_production_deployment.md)
   - [Role Registration](role_registration_guide.md)
   - [API Reference](13_api_reference.md)

4. **Contribute**
   - [Development Setup](../README.md#development)
   - [Testing Guide](11_testing_qa.md)
   - [Contributing Guidelines](../CONTRIBUTING.md)

---

**Last Updated:** 2025-01-15  
**Maintained by:** R3MES Development Team

