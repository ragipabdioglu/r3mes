"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";

interface MinerLocation {
  address: string;
  moniker: string;
  latitude: number;
  longitude: number;
  status: "active" | "inactive" | "jailed";
  reputation_score: number;
  country: string;
  city: string;
}

interface NetworkGlobeProps {
  showLabels?: boolean;
  autoRotate?: boolean;
  highlightActive?: boolean;
}

export default function NetworkGlobe({ 
  showLabels = true, 
  autoRotate = true,
  highlightActive = true 
}: NetworkGlobeProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredMiner, setHoveredMiner] = useState<MinerLocation | null>(null);
  const [rotation, setRotation] = useState({ x: 0, y: 0 });
  const isDragging = useRef(false);
  const lastMouse = useRef({ x: 0, y: 0 });

  // Fetch miner locations from API
  const { data: minerLocations } = useQuery<MinerLocation[]>({
    queryKey: ["miner-locations"],
    queryFn: async () => {
      const response = await fetch("/api/blockchain/miners/locations");
      if (!response.ok) {
        // Return mock data if API not available
        return generateMockLocations();
      }
      return response.json();
    },
    refetchInterval: 60000, // Refresh every minute
  });

  // Generate mock locations for demo
  const generateMockLocations = (): MinerLocation[] => {
    const cities = [
      { city: "New York", country: "USA", lat: 40.7128, lon: -74.0060 },
      { city: "London", country: "UK", lat: 51.5074, lon: -0.1278 },
      { city: "Tokyo", country: "Japan", lat: 35.6762, lon: 139.6503 },
      { city: "Singapore", country: "Singapore", lat: 1.3521, lon: 103.8198 },
      { city: "Frankfurt", country: "Germany", lat: 50.1109, lon: 8.6821 },
      { city: "Sydney", country: "Australia", lat: -33.8688, lon: 151.2093 },
      { city: "São Paulo", country: "Brazil", lat: -23.5505, lon: -46.6333 },
      { city: "Dubai", country: "UAE", lat: 25.2048, lon: 55.2708 },
      { city: "Seoul", country: "South Korea", lat: 37.5665, lon: 126.9780 },
      { city: "Mumbai", country: "India", lat: 19.0760, lon: 72.8777 },
      { city: "Toronto", country: "Canada", lat: 43.6532, lon: -79.3832 },
      { city: "Amsterdam", country: "Netherlands", lat: 52.3676, lon: 4.9041 },
    ];

    return cities.map((loc, i) => ({
      address: `remes1${Math.random().toString(36).substring(2, 15)}`,
      moniker: `Miner-${loc.city.replace(/\s/g, "")}`,
      latitude: loc.lat,
      longitude: loc.lon,
      status: Math.random() > 0.2 ? "active" : "inactive",
      reputation_score: Math.floor(Math.random() * 40) + 60,
      country: loc.country,
      city: loc.city,
    }));
  };

  // Convert lat/lon to 3D coordinates
  const latLonTo3D = useCallback((lat: number, lon: number, radius: number) => {
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lon + 180) * (Math.PI / 180);
    
    return {
      x: -(radius * Math.sin(phi) * Math.cos(theta)),
      y: radius * Math.cos(phi),
      z: radius * Math.sin(phi) * Math.sin(theta),
    };
  }, []);

  // Project 3D to 2D with rotation
  const project3D = useCallback((point: { x: number; y: number; z: number }, centerX: number, centerY: number, rotX: number, rotY: number) => {
    // Rotate around Y axis
    const cosY = Math.cos(rotY);
    const sinY = Math.sin(rotY);
    const x1 = point.x * cosY - point.z * sinY;
    const z1 = point.x * sinY + point.z * cosY;
    
    // Rotate around X axis
    const cosX = Math.cos(rotX);
    const sinX = Math.sin(rotX);
    const y1 = point.y * cosX - z1 * sinX;
    const z2 = point.y * sinX + z1 * cosX;
    
    // Simple perspective projection
    const scale = 300 / (300 + z2);
    
    return {
      x: centerX + x1 * scale,
      y: centerY + y1 * scale,
      z: z2,
      scale,
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let autoRotationAngle = 0;

    const resize = () => {
      canvas.width = canvas.offsetWidth * window.devicePixelRatio;
      canvas.height = canvas.offsetHeight * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };
    resize();
    window.addEventListener("resize", resize);

    const draw = () => {
      const width = canvas.offsetWidth;
      const height = canvas.offsetHeight;
      ctx.clearRect(0, 0, width, height);
      
      const centerX = width / 2;
      const centerY = height / 2;
      const radius = Math.min(width, height) * 0.35;
      
      const rotX = rotation.x;
      const rotY = rotation.y + (autoRotate ? autoRotationAngle : 0);

      // Draw globe wireframe
      ctx.strokeStyle = "#00ff41";
      ctx.lineWidth = 0.5;
      ctx.globalAlpha = 0.3;

      // Latitude lines
      const latSegments = 12;
      const lonSegments = 24;
      
      for (let i = 1; i < latSegments; i++) {
        const lat = -90 + (180 * i) / latSegments;
        ctx.beginPath();
        for (let j = 0; j <= lonSegments; j++) {
          const lon = -180 + (360 * j) / lonSegments;
          const point3D = latLonTo3D(lat, lon, radius);
          const projected = project3D(point3D, centerX, centerY, rotX, rotY);
          
          if (projected.z > 0) {
            if (j === 0) ctx.moveTo(projected.x, projected.y);
            else ctx.lineTo(projected.x, projected.y);
          }
        }
        ctx.stroke();
      }

      // Longitude lines
      for (let j = 0; j < lonSegments; j++) {
        const lon = -180 + (360 * j) / lonSegments;
        ctx.beginPath();
        for (let i = 0; i <= latSegments; i++) {
          const lat = -90 + (180 * i) / latSegments;
          const point3D = latLonTo3D(lat, lon, radius);
          const projected = project3D(point3D, centerX, centerY, rotX, rotY);
          
          if (projected.z > 0) {
            if (i === 0) ctx.moveTo(projected.x, projected.y);
            else ctx.lineTo(projected.x, projected.y);
          }
        }
        ctx.stroke();
      }

      // Draw miner locations
      ctx.globalAlpha = 1;
      const locations = minerLocations || [];
      
      // Sort by z-depth for proper rendering
      const projectedMiners = locations.map(miner => {
        const point3D = latLonTo3D(miner.latitude, miner.longitude, radius);
        const projected = project3D(point3D, centerX, centerY, rotX, rotY);
        return { miner, projected };
      }).sort((a, b) => a.projected.z - b.projected.z);

      projectedMiners.forEach(({ miner, projected }) => {
        if (projected.z > -radius * 0.3) { // Only show front-facing points
          const dotSize = Math.max(3, 6 * projected.scale);
          const alpha = Math.max(0.3, Math.min(1, (projected.z + radius) / (2 * radius)));
          
          // Glow effect
          const gradient = ctx.createRadialGradient(
            projected.x, projected.y, 0,
            projected.x, projected.y, dotSize * 2
          );
          
          const color = miner.status === "active" 
            ? (highlightActive ? "#00ff41" : "#22c55e")
            : miner.status === "jailed" 
              ? "#ef4444" 
              : "#6b7280";
          
          gradient.addColorStop(0, color);
          gradient.addColorStop(0.5, color + "80");
          gradient.addColorStop(1, "transparent");
          
          ctx.globalAlpha = alpha;
          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(projected.x, projected.y, dotSize * 2, 0, Math.PI * 2);
          ctx.fill();
          
          // Core dot
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(projected.x, projected.y, dotSize, 0, Math.PI * 2);
          ctx.fill();

          // Label for active miners
          if (showLabels && miner.status === "active" && projected.z > 0) {
            ctx.globalAlpha = alpha * 0.8;
            ctx.fillStyle = "#ffffff";
            ctx.font = `${10 * projected.scale}px monospace`;
            ctx.fillText(miner.moniker, projected.x + dotSize + 4, projected.y + 3);
          }
        }
      });

      if (autoRotate && !isDragging.current) {
        autoRotationAngle += 0.003;
      }
      
      animationFrameId = requestAnimationFrame(draw);
    };

    // Mouse interaction handlers
    const handleMouseDown = (e: MouseEvent) => {
      isDragging.current = true;
      lastMouse.current = { x: e.clientX, y: e.clientY };
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging.current) {
        const deltaX = e.clientX - lastMouse.current.x;
        const deltaY = e.clientY - lastMouse.current.y;
        
        setRotation(prev => ({
          x: prev.x + deltaY * 0.005,
          y: prev.y + deltaX * 0.005,
        }));
        
        lastMouse.current = { x: e.clientX, y: e.clientY };
      }
    };

    const handleMouseUp = () => {
      isDragging.current = false;
    };

    canvas.addEventListener("mousedown", handleMouseDown);
    canvas.addEventListener("mousemove", handleMouseMove);
    canvas.addEventListener("mouseup", handleMouseUp);
    canvas.addEventListener("mouseleave", handleMouseUp);

    draw();

    return () => {
      window.removeEventListener("resize", resize);
      canvas.removeEventListener("mousedown", handleMouseDown);
      canvas.removeEventListener("mousemove", handleMouseMove);
      canvas.removeEventListener("mouseup", handleMouseUp);
      canvas.removeEventListener("mouseleave", handleMouseUp);
      cancelAnimationFrame(animationFrameId);
    };
  }, [minerLocations, rotation, autoRotate, showLabels, highlightActive, latLonTo3D, project3D]);

  return (
    <div className="network-globe-container relative w-full h-full">
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-grab active:cursor-grabbing"
        style={{ background: "transparent" }}
      />
      
      {/* Stats overlay */}
      <div className="absolute bottom-4 left-4 bg-black/60 backdrop-blur-sm rounded-lg p-3 text-xs">
        <div className="text-green-400 font-mono">
          <div>Active Miners: {minerLocations?.filter(m => m.status === "active").length || 0}</div>
          <div>Total Nodes: {minerLocations?.length || 0}</div>
          <div>Countries: {new Set(minerLocations?.map(m => m.country)).size || 0}</div>
        </div>
      </div>

      {/* Hovered miner info */}
      {hoveredMiner && (
        <div className="absolute top-4 right-4 bg-black/80 backdrop-blur-sm rounded-lg p-3 text-xs">
          <div className="text-white font-semibold">{hoveredMiner.moniker}</div>
          <div className="text-gray-400">{hoveredMiner.city}, {hoveredMiner.country}</div>
          <div className="text-green-400">Score: {hoveredMiner.reputation_score}</div>
        </div>
      )}
    </div>
  );
}
