"use client";

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { useState, useEffect } from "react";

interface TrainingMetrics {
  epoch: number;
  loss: number;
  accuracy: number;
  gradient_norm: number;
  timestamp: number;
}

interface TrainingGraphProps {
  metrics: TrainingMetrics | null;
}

export default function TrainingGraph({ metrics }: TrainingGraphProps) {
  const [data, setData] = useState<Array<{ epoch: number; loss: number; accuracy: number }>>([]);

  useEffect(() => {
    if (metrics) {
      setData((prev) => {
        const newData = [...prev, { epoch: metrics.epoch, loss: metrics.loss, accuracy: metrics.accuracy }];
        // Keep only last 50 data points
        return newData.slice(-50);
      });
    }
  }, [metrics]);

  if (data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-500 dark:text-gray-400">
        Waiting for training data...
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="epoch" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line
          type="monotone"
          dataKey="loss"
          stroke="#8884d8"
          strokeWidth={2}
          name="Loss"
          dot={false}
        />
        <Line
          type="monotone"
          dataKey="accuracy"
          stroke="#82ca9d"
          strokeWidth={2}
          name="Accuracy"
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

