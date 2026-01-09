"use client";

import { Cpu, Fan, HardDrive, Zap } from "lucide-react";

interface MinerStats {
  gpu_temp: number;
  fan_speed: number;
  vram_usage: number;
  power_draw: number;
  hash_rate: number;
  uptime: number;
  timestamp: number;
}

interface HardwareMonitorProps {
  stats: MinerStats | null;
}

export default function HardwareMonitor({ stats }: HardwareMonitorProps) {
  if (!stats) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-500 dark:text-gray-400">
        Waiting for hardware stats...
      </div>
    );
  }

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  return (
    <div className="grid grid-cols-2 gap-4">
      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
        <div className="flex items-center space-x-2 mb-2">
          <Cpu className="w-5 h-5 text-blue-600" />
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">GPU Temp</span>
        </div>
        <div className="text-2xl font-bold text-gray-900 dark:text-white">
          {stats.gpu_temp.toFixed(1)}°C
        </div>
      </div>

      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
        <div className="flex items-center space-x-2 mb-2">
          <Fan className="w-5 h-5 text-green-600" />
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Fan Speed</span>
        </div>
        <div className="text-2xl font-bold text-gray-900 dark:text-white">
          {stats.fan_speed.toLocaleString()} RPM
        </div>
      </div>

      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
        <div className="flex items-center space-x-2 mb-2">
          <HardDrive className="w-5 h-5 text-purple-600" />
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">VRAM Usage</span>
        </div>
        <div className="text-2xl font-bold text-gray-900 dark:text-white">
          {(stats.vram_usage / 1024).toFixed(1)} GB
        </div>
      </div>

      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
        <div className="flex items-center space-x-2 mb-2">
          <Zap className="w-5 h-5 text-yellow-600" />
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Power Draw</span>
        </div>
        <div className="text-2xl font-bold text-gray-900 dark:text-white">
          {stats.power_draw.toFixed(0)} W
        </div>
      </div>

      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 col-span-2">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Hash Rate
            </div>
            <div className="text-xl font-bold text-gray-900 dark:text-white">
              {stats.hash_rate.toFixed(2)} gradients/hour
            </div>
          </div>
          <div>
            <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Uptime
            </div>
            <div className="text-xl font-bold text-gray-900 dark:text-white">
              {formatUptime(stats.uptime)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

