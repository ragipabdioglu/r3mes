"use client";

import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Mesh, MeshStandardMaterial } from "three";
import * as THREE from "three";

export default function BlockchainVisualization() {
  const chainRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (chainRef.current) {
      chainRef.current.rotation.y = state.clock.elapsedTime * 0.1;
    }
  });

  const blocks = Array.from({ length: 12 }, (_, i) => {
    const angle = (i / 12) * Math.PI * 2;
    const radius = 2.5;
    return {
      position: [
        Math.cos(angle) * radius,
        Math.sin(angle) * radius * 0.3,
        Math.sin(angle) * radius * 0.5,
      ] as [number, number, number],
    };
  });

  return (
    <group ref={chainRef}>
      {blocks.map((block, i) => (
        <Block key={i} position={block.position} delay={i * 0.1} />
      ))}
    </group>
  );
}

function Block({
  position,
  delay,
}: {
  position: [number, number, number];
  delay: number;
}) {
  const meshRef = useRef<Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      const time = state.clock.elapsedTime + delay;
      const material = meshRef.current.material as MeshStandardMaterial;
      if (material && 'emissiveIntensity' in material) {
        material.emissiveIntensity = 0.3 + Math.sin(time * 2) * 0.2;
      }
    }
  });

  return (
    <mesh ref={meshRef} position={position}>
      <boxGeometry args={[0.2, 0.2, 0.2]} />
      <meshStandardMaterial
        color="#0071e3"
        emissive="#0071e3"
        emissiveIntensity={0.15}
        transparent
        opacity={0.25}
      />
    </mesh>
  );
}

