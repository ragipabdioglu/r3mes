"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle, ArrowRight, Zap, Shield, Database, Network } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";

const steps = [
  {
    id: 1,
    title: "Hoşgeldiniz!",
    description: "R3MES'e hoş geldiniz. Sisteminiz analiz ediliyor...",
    icon: <Zap className="w-12 h-12" />,
  },
  {
    id: 2,
    title: "Cüzdan Bağlantısı",
    description: "Keplr cüzdanınızı bağlayarak başlayın",
    icon: <Shield className="w-12 h-12" />,
  },
  {
    id: 3,
    title: "Madencilik",
    description: "GPU'nuzu bağlayın ve REMES kazanmaya başlayın",
    icon: <Database className="w-12 h-12" />,
  },
  {
    id: 4,
    title: "Hazırsınız!",
    description: "Artık R3MES'i kullanmaya başlayabilirsiniz",
    icon: <Network className="w-12 h-12" />,
  },
];

export default function OnboardingPage() {
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState(0);
  const [completed, setCompleted] = useState(false);

  useEffect(() => {
    // Check if onboarding was already completed
    const onboardingCompleted = localStorage.getItem("r3mes_onboarding_completed");
    if (onboardingCompleted === "true") {
      router.push("/");
      return;
    }

    // Auto-advance steps
    const timer = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev < steps.length - 1) {
          return prev + 1;
        } else {
          clearInterval(timer);
          setCompleted(true);
          return prev;
        }
      });
    }, 2000);

    return () => clearInterval(timer);
  }, [router]);

  const handleComplete = () => {
    localStorage.setItem("r3mes_onboarding_completed", "true");
    router.push("/");
  };

  if (completed) {
    return (
      <div className="min-h-screen bg-slate-900 text-slate-100 flex items-center justify-center px-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center max-w-2xl"
        >
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: "spring" }}
            className="mb-8"
          >
            <CheckCircle className="w-24 h-24 text-primary mx-auto" />
          </motion.div>
          <h1 className="text-4xl font-bold mb-4 gradient-text">
            Hazırsınız!
          </h1>
          <p className="text-xl text-slate-400 mb-8">
            R3MES'e hoş geldiniz. Artık başlayabilirsiniz!
          </p>
          <Link
            href="/"
            className="btn-primary text-lg px-8 py-4 inline-flex items-center gap-2"
          >
            Başlayalım
            <ArrowRight className="w-5 h-5" />
          </Link>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex items-center justify-center px-4">
      <div className="max-w-4xl w-full">
        {/* Progress Bar */}
        <div className="mb-12">
          <div className="flex items-center justify-between mb-4">
            {steps.map((step, index) => (
              <div
                key={step.id}
                className={`flex-1 flex items-center ${
                  index < steps.length - 1 ? "mr-2" : ""
                }`}
              >
                <div className="flex-1 flex items-center">
                  <div
                    className={`w-12 h-12 rounded-full flex items-center justify-center border-2 transition-colors ${
                      index <= currentStep
                        ? "bg-primary border-primary text-slate-900"
                        : "bg-slate-800 border-slate-700 text-slate-400"
                    }`}
                  >
                    {index < currentStep ? (
                      <CheckCircle className="w-6 h-6" />
                    ) : (
                      step.id
                    )}
                  </div>
                  {index < steps.length - 1 && (
                    <div
                      className={`flex-1 h-1 mx-2 transition-colors ${
                        index < currentStep ? "bg-primary" : "bg-slate-700"
                      }`}
                    />
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Current Step Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={currentStep}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="text-center"
          >
            <div className="mb-8 flex justify-center">
              <div className="text-primary">{steps[currentStep].icon}</div>
            </div>
            <h1 className="text-4xl font-bold mb-4 gradient-text">
              {steps[currentStep].title}
            </h1>
            <p className="text-xl text-slate-400 mb-8">
              {steps[currentStep].description}
            </p>
          </motion.div>
        </AnimatePresence>

        {/* Skip Button */}
        <div className="text-center">
          <button
            onClick={handleComplete}
            className="text-slate-400 hover:text-slate-300 transition-colors"
          >
            Atla ve devam et →
          </button>
        </div>
      </div>
    </div>
  );
}

