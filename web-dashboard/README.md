# R3MES Web Dashboard

Professional monitoring interface for R3MES Proof of Useful Work (PoUW) miners.

## Features

- **Miner Console**: Real-time training metrics and hardware monitoring (Zero-GPU interface)
- **Network Explorer**: Global view of network nodes and statistics
- **WebSocket Streaming**: Real-time data updates
- **Keplr Wallet Integration**: Connect your Cosmos wallet

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- R3MES blockchain node running on `localhost:1317`

### Installation

```bash
npm install
```

### Development

**Important**: If you're using WSL, run the commands from within WSL, not from Windows:

```bash
# From WSL terminal
cd /home/rabdi/R3MES/web-dashboard
npm run dev
```

If you encounter UNC path errors when running from Windows, use WSL:

```bash
# In WSL
wsl
cd ~/R3MES/web-dashboard
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build

```bash
npm run build
npm start
```

## Architecture

- **Next.js 14**: React framework with TypeScript
- **Tailwind CSS**: Utility-first styling
- **Recharts**: 2D charting (zero GPU usage)
- **TanStack Query**: State management and caching
- **WebSocket**: Real-time data streaming

## Critical Requirements

- **Zero GPU Usage**: Miner Console uses strictly 2D components (Recharts)
- **Network Explorer**: Uses lazy-loaded 3D globe (react-globe.gl) only for network visualization
- **Real-time Updates**: WebSocket connections for live data

## Troubleshooting

### UNC Path Error

If you see "UNC paths are not supported" error, you're running from Windows. Use WSL instead:

```bash
wsl
cd ~/R3MES/web-dashboard
npm run dev
```

### App Directory Not Found

Make sure you're in the correct directory:
```bash
pwd  # Should show: /home/rabdi/R3MES/web-dashboard
ls app/  # Should show: layout.tsx, page.tsx, globals.css
```
