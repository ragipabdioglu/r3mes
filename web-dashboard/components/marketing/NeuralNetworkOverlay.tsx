"use client";

import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { MeshStandardMaterial } from "three";

export default function NeuralNetworkOverlay() {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.z = state.clock.elapsedTime * 0.02;
    }
  });

  const layers = 3;
  const nodesPerLayer = 8;

  return (
    <group ref={groupRef}>
      {Array.from({ length: layers }, (_, layerIndex) => (
        <NeuralLayer
          key={layerIndex}
          layerIndex={layerIndex}
          nodesPerLayer={nodesPerLayer}
          totalLayers={layers}
        />
      ))}
    </group>
  );
}

function NeuralLayer({
  layerIndex,
  nodesPerLayer,
  totalLayers,
}: {
  layerIndex: number;
  nodesPerLayer: number;
  totalLayers: number;
}) {
  const radius = 2;
  const yPosition = (layerIndex - (totalLayers - 1) / 2) * 0.8;

  return (
    <group position={[0, yPosition, 0]}>
      {Array.from({ length: nodesPerLayer }, (_, nodeIndex) => {
        const angle = (nodeIndex / nodesPerLayer) * Math.PI * 2;
        const x = Math.cos(angle) * radius;
        const z = Math.sin(angle) * radius;
        return (
          <NeuralNode
            key={nodeIndex}
            position={[x, 0, z]}
            layerIndex={layerIndex}
            nodeIndex={nodeIndex}
          />
        );
      })}
    </group>
  );
}

function NeuralNode({
  position,
  layerIndex,
  nodeIndex,
}: {
  position: [number, number, number];
  layerIndex: number;
  nodeIndex: number;
}) {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      const time = state.clock.elapsedTime;
      const pulse = Math.sin(time * 2 + layerIndex + nodeIndex) * 0.1 + 0.2;
      const material = meshRef.current.material as MeshStandardMaterial;
      if (material && 'emissiveIntensity' in material) {
        material.emissiveIntensity = pulse;
      }
    }
  });

  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[0.05, 16, 16]} />
      <meshStandardMaterial
        color="#0071e3"
        emissive="#0071e3"
        emissiveIntensity={0.1}
        transparent
        opacity={0.2}
      />
    </mesh>
  );
}

