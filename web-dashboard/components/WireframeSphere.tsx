"use client";

import { useEffect, useRef } from "react";

export default function WireframeSphere() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let rotation = 0;

    const resize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const radius = Math.min(canvas.width, canvas.height) * 0.3;
      
      ctx.strokeStyle = "#00ff41";
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.6;

      // Draw wireframe sphere (simplified)
      const segments = 20;
      for (let i = 0; i <= segments; i++) {
        const lat = (Math.PI * i) / segments;
        const y = centerY + radius * Math.cos(lat);
        const r = radius * Math.sin(lat);

        ctx.beginPath();
        for (let j = 0; j <= segments; j++) {
          const lon = (2 * Math.PI * j) / segments + rotation;
          const x = centerX + r * Math.cos(lon);
          if (j === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      }

      // Draw vertical lines
      for (let j = 0; j <= segments; j++) {
        const lon = (2 * Math.PI * j) / segments + rotation;
        ctx.beginPath();
        for (let i = 0; i <= segments; i++) {
          const lat = (Math.PI * i) / segments;
          const y = centerY + radius * Math.cos(lat);
          const r = radius * Math.sin(lat);
          const x = centerX + r * Math.cos(lon);
          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      }

      rotation += 0.01;
      animationFrameId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full"
      style={{ opacity: 0.3 }}
    />
  );
}

