"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, Float } from "@react-three/drei";
import { Suspense } from "react";
import Globe3D from "./Globe3D";

export default function HeroScene() {
  return (
    <Canvas
      camera={{ position: [0, 0, 6], fov: 60 }}
      gl={{ 
        antialias: true, 
        alpha: true,
        powerPreference: "high-performance"
      }}
      className="w-full h-full"
      dpr={[1, 2]}
    >
      <Suspense fallback={null}>
        {/* Ambient lighting */}
        <ambientLight intensity={0.2} />
        
        {/* Key light */}
        <pointLight position={[10, 10, 10]} intensity={0.4} color="#ffffff" />
        
        {/* Fill light */}
        <pointLight position={[-10, -10, -10]} intensity={0.2} color="#0071e3" />
        
        {/* Rim light for glow effect */}
        <pointLight position={[0, 0, -10]} intensity={0.3} color="#0071e3" />
        
        {/* Floating animation wrapper */}
        <Float
          speed={1.5}
          rotationIntensity={0.2}
          floatIntensity={0.3}
        >
          <Globe3D />
        </Float>
        
        {/* Subtle orbit controls */}
        <OrbitControls
          enableZoom={false}
          enablePan={false}
          autoRotate
          autoRotateSpeed={0.2}
          minPolarAngle={Math.PI / 3}
          maxPolarAngle={Math.PI / 1.5}
          enableDamping
          dampingFactor={0.05}
        />
      </Suspense>
    </Canvas>
  );
}
