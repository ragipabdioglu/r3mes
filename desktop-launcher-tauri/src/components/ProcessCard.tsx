import "./ProcessCard.css";

interface ProcessCardProps {
  name: string;
  status: "running" | "stopped";
  pid: number | null;
  onStart: () => void;
  onStop: () => void;
}

export default function ProcessCard({
  name,
  status,
  pid,
  onStart,
  onStop,
}: ProcessCardProps) {
  return (
    <div className={`process-card ${status}`}>
      <div className="process-header">
        <h3>{name}</h3>
        <div className={`status-indicator ${status}`}>
          <span className="status-dot"></span>
          <span className="status-text">
            {status === "running" ? "Running" : "Stopped"}
          </span>
        </div>
      </div>
      {pid && status === "running" && (
        <div className="process-info">
          <span className="pid">PID: {pid}</span>
        </div>
      )}
      <div className="process-actions">
        {status === "running" ? (
          <button className="btn-stop" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button className="btn-start" onClick={onStart}>
            Start
          </button>
        )}
      </div>
    </div>
  );
}

