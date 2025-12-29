import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/tauri";
import "./SetupWizard.css";

interface HardwareCheckResult {
  docker: {
    installed: boolean;
    running: boolean;
    version: string | null;
  };
  gpu: {
    available: boolean;
    name: string | null;
    vram_gb: number | null;
    driver_version: string | null;
  };
  disk: {
    available_gb: number;
    required_gb: number;
    sufficient: boolean;
  };
  ram: {
    total_gb: number;
    minimum_gb: number;
    sufficient: boolean;
  };
  cuda: {
    installed: boolean;
    version: string | null;
    compatible: boolean;
  };
  all_checks_passed: boolean;
}

interface SetupWizardProps {
  onComplete: () => void;
}

export default function SetupWizard({ onComplete }: SetupWizardProps) {
  const [step, setStep] = useState(1);
  const [checking, setChecking] = useState(false);
  const [hardwareResult, setHardwareResult] = useState<HardwareCheckResult | null>(null);
  const [selectedRoles, setSelectedRoles] = useState<number[]>([1]); // Default: Miner

  useEffect(() => {
    runHardwareCheck();
  }, []);

  const runHardwareCheck = async () => {
    setChecking(true);
    try {
      const result = await invoke<HardwareCheckResult>("check_hardware");
      setHardwareResult(result);
    } catch (error) {
      console.error("Hardware check failed:", error);
    } finally {
      setChecking(false);
    }
  };

  const handleComplete = async () => {
    try {
      await invoke("mark_setup_complete");
      onComplete();
    } catch (error) {
      console.error("Failed to mark setup complete:", error);
    }
  };

  const getDockerInstallLink = () => {
    const platform = navigator.platform.toLowerCase();
    if (platform.includes("win")) {
      return "https://www.docker.com/products/docker-desktop/";
    } else if (platform.includes("mac")) {
      return "https://www.docker.com/products/docker-desktop/";
    } else {
      return "https://docs.docker.com/engine/install/";
    }
  };

  if (checking) {
    return (
      <div className="setup-wizard">
        <div className="wizard-content">
          <h2>Checking System Requirements...</h2>
          <div className="loading-spinner"></div>
        </div>
      </div>
    );
  }

  if (!hardwareResult) {
    return (
      <div className="setup-wizard">
        <div className="wizard-content">
          <h2>Setup Wizard</h2>
          <p>Unable to check hardware. Please try again.</p>
          <button onClick={runHardwareCheck}>Retry</button>
        </div>
      </div>
    );
  }

  return (
    <div className="setup-wizard">
      <div className="wizard-content">
        <div className="wizard-header">
          <h1>Welcome to R3MES Launcher</h1>
          <p className="subtitle">Let's check if your system is ready for mining</p>
        </div>

        <div className="wizard-steps">
          <div className={`step ${step >= 1 ? "active" : ""}`}>
            <div className="step-number">1</div>
            <div className="step-title">Hardware Check</div>
          </div>
          <div className={`step ${step >= 2 ? "active" : ""}`}>
            <div className="step-number">2</div>
            <div className="step-title">Role Selection</div>
          </div>
          <div className={`step ${step >= 3 ? "active" : ""}`}>
            <div className="step-number">3</div>
            <div className="step-title">Review</div>
          </div>
          <div className={`step ${step >= 4 ? "active" : ""}`}>
            <div className="step-number">4</div>
            <div className="step-title">Complete</div>
          </div>
        </div>

        {step === 1 && (
          <div className="wizard-step-content">
            <h2>Hardware Requirements Check</h2>
            
            <div className="check-list">
              {/* Docker Check */}
              <div className={`check-item ${hardwareResult.docker.installed ? "pass" : "fail"}`}>
                <div className="check-icon">
                  {hardwareResult.docker.installed ? "✅" : "❌"}
                </div>
                <div className="check-details">
                  <div className="check-name">Docker</div>
                  <div className="check-status">
                    {hardwareResult.docker.installed ? (
                      <>
                        Installed {hardwareResult.docker.version && `(${hardwareResult.docker.version})`}
                        {hardwareResult.docker.running ? " - Running" : " - Not Running"}
                      </>
                    ) : (
                      <>
                        Not Installed
                        <a href={getDockerInstallLink()} target="_blank" rel="noopener noreferrer" className="install-link">
                          Install Docker →
                        </a>
                      </>
                    )}
                  </div>
                </div>
              </div>

              {/* GPU Check */}
              <div className={`check-item ${hardwareResult.gpu.available ? "pass" : "fail"}`}>
                <div className="check-icon">
                  {hardwareResult.gpu.available ? "✅" : "❌"}
                </div>
                <div className="check-details">
                  <div className="check-name">NVIDIA GPU</div>
                  <div className="check-status">
                    {hardwareResult.gpu.available ? (
                      <>
                        {hardwareResult.gpu.name} ({hardwareResult.gpu.vram_gb}GB VRAM)
                        {hardwareResult.gpu.driver_version && ` - Driver: ${hardwareResult.gpu.driver_version}`}
                      </>
                    ) : (
                      "No NVIDIA GPU detected"
                    )}
                  </div>
                </div>
              </div>

              {/* CUDA Check */}
              <div className={`check-item ${hardwareResult.cuda.compatible ? "pass" : "fail"}`}>
                <div className="check-icon">
                  {hardwareResult.cuda.compatible ? "✅" : "❌"}
                </div>
                <div className="check-details">
                  <div className="check-name">CUDA (≥12.1)</div>
                  <div className="check-status">
                    {hardwareResult.cuda.installed ? (
                      hardwareResult.cuda.compatible ? (
                        `Installed (${hardwareResult.cuda.version}) - Compatible`
                      ) : (
                        `Installed (${hardwareResult.cuda.version}) - Version too old, need ≥12.1`
                      )
                    ) : (
                      "Not Installed"
                    )}
                  </div>
                </div>
              </div>

              {/* Disk Check */}
              <div className={`check-item ${hardwareResult.disk.sufficient ? "pass" : "fail"}`}>
                <div className="check-icon">
                  {hardwareResult.disk.sufficient ? "✅" : "⚠️"}
                </div>
                <div className="check-details">
                  <div className="check-name">Disk Space</div>
                  <div className="check-status">
                    {hardwareResult.disk.available_gb}GB available / {hardwareResult.disk.required_gb}GB required
                    {!hardwareResult.disk.sufficient && (
                      <span className="warning"> - Insufficient space</span>
                    )}
                  </div>
                </div>
              </div>

              {/* RAM Check */}
              <div className={`check-item ${hardwareResult.ram.sufficient ? "pass" : "fail"}`}>
                <div className="check-icon">
                  {hardwareResult.ram.sufficient ? "✅" : "⚠️"}
                </div>
                <div className="check-details">
                  <div className="check-name">RAM</div>
                  <div className="check-status">
                    {hardwareResult.ram.total_gb}GB / {hardwareResult.ram.minimum_gb}GB minimum
                    {!hardwareResult.ram.sufficient && (
                      <span className="warning"> - Below minimum</span>
                    )}
                  </div>
                </div>
              </div>
            </div>

            <div className="wizard-actions">
              <button onClick={runHardwareCheck} className="btn-secondary">
                Re-check
              </button>
              {!hardwareResult.all_checks_passed && (
                <button
                  onClick={async () => {
                    // Try to install missing components
                    const missingComponents = [];
                    if (!hardwareResult.docker.installed) missingComponents.push("docker");
                    if (!hardwareResult.cuda.installed) missingComponents.push("cuda");
                    
                    for (const component of missingComponents) {
                      try {
                        await invoke("install_component", { component });
                      } catch (error) {
                        console.error(`Failed to install ${component}:`, error);
                      }
                    }
                    
                    // Re-check after installation attempts
                    await runHardwareCheck();
                  }}
                  className="btn-primary"
                >
                  Install Missing Components
                </button>
              )}
              <button
                onClick={() => setStep(2)}
                className="btn-primary"
                disabled={!hardwareResult.all_checks_passed}
              >
                Continue
              </button>
        )}

        {step === 2 && (
          <div className="wizard-step-content">
            <h2>Select Node Roles</h2>
            <p>Choose which roles your node will perform. You can select multiple roles.</p>
            
            <div className="role-selection">
              <div
                className={`role-card ${selectedRoles.includes(1) ? "selected" : ""}`}
                onClick={() => {
                  if (selectedRoles.includes(1)) {
                    setSelectedRoles(selectedRoles.filter(r => r !== 1));
                  } else {
                    setSelectedRoles([...selectedRoles, 1]);
                  }
                }}
              >
                <div className="role-icon">⚡</div>
                <div className="role-name">Miner</div>
                <div className="role-description">AI model training and gradient submission</div>
                <div className="role-requirements">Requires: GPU, CUDA</div>
              </div>
              
              <div
                className={`role-card ${selectedRoles.includes(2) ? "selected" : ""}`}
                onClick={() => {
                  if (selectedRoles.includes(2)) {
                    setSelectedRoles(selectedRoles.filter(r => r !== 2));
                  } else {
                    setSelectedRoles([...selectedRoles, 2]);
                  }
                }}
              >
                <div className="role-icon">🖥️</div>
                <div className="role-name">Serving</div>
                <div className="role-description">AI model inference serving</div>
                <div className="role-requirements">Requires: GPU/CPU, IPFS</div>
              </div>
              
              <div
                className={`role-card ${selectedRoles.includes(3) ? "selected" : ""}`}
                onClick={() => {
                  if (selectedRoles.includes(3)) {
                    setSelectedRoles(selectedRoles.filter(r => r !== 3));
                  } else {
                    setSelectedRoles([...selectedRoles, 3]);
                  }
                }}
              >
                <div className="role-icon">🛡️</div>
                <div className="role-name">Validator</div>
                <div className="role-description">Blockchain validation and consensus</div>
                <div className="role-requirements">Requires: Stable connection, stake</div>
              </div>
              
              <div
                className={`role-card ${selectedRoles.includes(4) ? "selected" : ""}`}
                onClick={() => {
                  if (selectedRoles.includes(4)) {
                    setSelectedRoles(selectedRoles.filter(r => r !== 4));
                  } else {
                    setSelectedRoles([...selectedRoles, 4]);
                  }
                }}
              >
                <div className="role-icon">🔗</div>
                <div className="role-name">Proposer</div>
                <div className="role-description">Gradient aggregation and submission</div>
                <div className="role-requirements">Requires: CPU, IPFS, stake</div>
              </div>
            </div>

            <div className="wizard-actions">
              <button onClick={() => setStep(1)} className="btn-secondary">
                Back
              </button>
              <button
                onClick={() => setStep(3)}
                className="btn-primary"
                disabled={selectedRoles.length === 0}
              >
                Continue
              </button>
            </div>
          </div>
            </div>
          </div>
        )}

        {step === 3 && (
          <div className="wizard-step-content">
            <h2>Review & Complete</h2>
            <p>All system requirements have been checked. You're ready to start!</p>
            
            <div className="review-section">
              <h3>Selected Roles:</h3>
              <div className="selected-roles">
                {selectedRoles.map(roleId => {
                  const roleNames: Record<number, string> = {
                    1: "Miner",
                    2: "Serving",
                    3: "Validator",
                    4: "Proposer",
                  };
                  return (
                    <span key={roleId} className="role-badge">
                      {roleNames[roleId]}
                    </span>
                  );
                })}
              </div>
            </div>
            
            {!hardwareResult.all_checks_passed && (
              <div className="warning-box">
                <strong>⚠️ Warning:</strong> Some requirements are not met. Some roles may not work properly.
                You can still continue, but performance may be affected.
              </div>
            )}

            <div className="info-box" style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#f0f0f0', borderRadius: '8px' }}>
              <strong>💡 Next Step:</strong> After completing setup, register your node roles on the blockchain via the web dashboard at <code>/roles</code> or use the blockchain CLI. See the documentation for detailed registration instructions.
            </div>

            <div className="wizard-actions">
              <button onClick={() => setStep(2)} className="btn-secondary">
                Back
              </button>
              <button onClick={handleComplete} className="btn-primary">
                Complete Setup
              </button>
            </div>
          </div>
        )}

        {step === 4 && (
          <div className="wizard-step-content">
            <h2>Setup Complete!</h2>
            <p>Your R3MES node is configured and ready to use.</p>
            <div className="wizard-actions">
              <button onClick={handleComplete} className="btn-primary">
                Get Started
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

