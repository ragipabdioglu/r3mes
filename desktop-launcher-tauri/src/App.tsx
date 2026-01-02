import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/tauri";
import ProcessCard from "./components/ProcessCard";
import LogViewer from "./components/LogViewer";
import SetupWizard from "./components/SetupWizard";
import WalletManager from "./components/WalletManager";
import SystemStatusPanel from "./components/SystemStatusPanel";
import MiningDashboard from "./components/MiningDashboard";
import "./App.css";

interface ProcessStatus {
  node: { running: boolean; pid: number | null };
  miner: { running: boolean; pid: number | null };
  ipfs: { running: boolean; pid: number | null };
  serving: { running: boolean; pid: number | null };
  validator: { running: boolean; pid: number | null };
  proposer: { running: boolean; pid: number | null };
}

function App() {
  const [showSetupWizard, setShowSetupWizard] = useState(false);
  const [showWalletManager, setShowWalletManager] = useState(false);
  const [status, setStatus] = useState<ProcessStatus>({
    node: { running: false, pid: null },
    miner: { running: false, pid: null },
    ipfs: { running: false, pid: null },
    serving: { running: false, pid: null },
    validator: { running: false, pid: null },
    proposer: { running: false, pid: null },
  });
  const [selectedProcess, setSelectedProcess] = useState<string>("node");

  useEffect(() => {
    // Check if this is first run
    const checkFirstRun = async () => {
      try {
        const isFirstRun = await invoke<boolean>("is_first_run");
        setShowSetupWizard(isFirstRun);
      } catch (error) {
        console.error("Failed to check first run:", error);
        // If check fails, assume not first run
        setShowSetupWizard(false);
      }
    };
    checkFirstRun();
  }, []);

  useEffect(() => {
    // Poll status every 2 seconds
    const interval = setInterval(async () => {
      try {
        const currentStatus = await invoke<ProcessStatus>("get_status");
        setStatus(currentStatus);
      } catch (error) {
        console.error("Failed to get status:", error);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const handleStartNode = async () => {
    try {
      await invoke("start_node");
    } catch (error) {
      console.error("[Node] Error:", error);
    }
  };

  const handleStopNode = async () => {
    try {
      await invoke("stop_node");
    } catch (error) {
      console.error("[Node] Error:", error);
    }
  };

  const handleStartMiner = async () => {
    try {
      await invoke("start_miner");
    } catch (error) {
      console.error("[Miner] Error:", error);
    }
  };

  const handleStopMiner = async () => {
    try {
      await invoke("stop_miner");
    } catch (error) {
      console.error("[Miner] Error:", error);
    }
  };

  const handleStartIPFS = async () => {
    try {
      await invoke("start_ipfs");
    } catch (error) {
      console.error("[IPFS] Error:", error);
    }
  };

  const handleStopIPFS = async () => {
    try {
      await invoke("stop_ipfs");
    } catch (error) {
      console.error("[IPFS] Error:", error);
    }
  };

  const handleStartServing = async () => {
    try {
      await invoke("start_serving");
    } catch (error) {
      console.error("[Serving] Error:", error);
    }
  };

  const handleStopServing = async () => {
    try {
      await invoke("stop_serving");
    } catch (error) {
      console.error("[Serving] Error:", error);
    }
  };

  const handleStartValidator = async () => {
    try {
      await invoke("start_validator");
    } catch (error) {
      console.error("[Validator] Error:", error);
    }
  };

  const handleStopValidator = async () => {
    try {
      await invoke("stop_validator");
    } catch (error) {
      console.error("[Validator] Error:", error);
    }
  };

  const handleStartProposer = async () => {
    try {
      await invoke("start_proposer");
    } catch (error) {
      console.error("[Proposer] Error:", error);
    }
  };

  const handleStopProposer = async () => {
    try {
      await invoke("stop_proposer");
    } catch (error) {
      console.error("[Proposer] Error:", error);
    }
  };


  if (showSetupWizard) {
    return (
      <SetupWizard
        onComplete={() => {
          setShowSetupWizard(false);
        }}
      />
    );
  }

  return (
    <div className="app">
      {showWalletManager && (
        <WalletManager onClose={() => setShowWalletManager(false)} />
      )}
      
      <header className="app-header">
        <h1>R3MES Launcher</h1>
        <p className="subtitle">Native Desktop Control Panel</p>
        <div style={{ position: "absolute", right: "20px", top: "50%", transform: "translateY(-50%)", display: "flex", gap: "8px" }}>
          <button
            onClick={async () => {
              try {
                await invoke("open_dashboard");
              } catch (error) {
                console.error("Failed to open dashboard:", error);
                // Fallback: open in new window
                window.open("http://localhost:3000", "_blank");
              }
            }}
            className="wallet-btn"
            style={{
              padding: "8px 16px",
              background: "#06b6d4",
              color: "#ffffff",
              border: "none",
              borderRadius: "8px",
              cursor: "pointer",
              fontSize: "14px",
              fontWeight: "500",
            }}
          >
            🌐 Open Dashboard
          </button>
          <button
            onClick={() => setShowWalletManager(true)}
            className="wallet-btn"
            style={{
              padding: "8px 16px",
              background: "#3b82f6",
              color: "#ffffff",
              border: "none",
              borderRadius: "8px",
              cursor: "pointer",
              fontSize: "14px",
              fontWeight: "500",
            }}
          >
            💼 Wallet
          </button>
        </div>
      </header>

      <main className="app-main">
        {status.miner.running && <MiningDashboard />}
        
        <div className="process-grid">
          <ProcessCard
            name="Blockchain Node"
            status={status.node.running ? "running" : "stopped"}
            pid={status.node.pid}
            onStart={handleStartNode}
            onStop={handleStopNode}
          />
          <ProcessCard
            name="Miner"
            status={status.miner.running ? "running" : "stopped"}
            pid={status.miner.pid}
            onStart={handleStartMiner}
            onStop={handleStopMiner}
          />
          <ProcessCard
            name="IPFS"
            status={status.ipfs.running ? "running" : "stopped"}
            pid={status.ipfs.pid}
            onStart={handleStartIPFS}
            onStop={handleStopIPFS}
          />
          <ProcessCard
            name="Serving Node"
            status={status.serving.running ? "running" : "stopped"}
            pid={status.serving.pid}
            onStart={handleStartServing}
            onStop={handleStopServing}
          />
          <ProcessCard
            name="Validator"
            status={status.validator.running ? "running" : "stopped"}
            pid={status.validator.pid}
            onStart={handleStartValidator}
            onStop={handleStopValidator}
          />
          <ProcessCard
            name="Proposer"
            status={status.proposer.running ? "running" : "stopped"}
            pid={status.proposer.pid}
            onStart={handleStartProposer}
            onStop={handleStopProposer}
          />
        </div>

        <div className="status-and-logs">
          <SystemStatusPanel />
          
          <div className="logs-section">
            <h2 className="section-title">Process Logs</h2>
            <div className="log-tabs">
              <button
                onClick={() => setSelectedProcess("node")}
                className={selectedProcess === "node" ? "active" : ""}
              >
                Node
              </button>
              <button
                onClick={() => setSelectedProcess("miner")}
                className={selectedProcess === "miner" ? "active" : ""}
              >
                Miner
              </button>
              <button
                onClick={() => setSelectedProcess("ipfs")}
                className={selectedProcess === "ipfs" ? "active" : ""}
              >
                IPFS
              </button>
            </div>
            <LogViewer process={selectedProcess} />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;

