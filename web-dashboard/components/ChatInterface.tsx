"use client";

import { useState, useRef, useEffect } from "react";
import { sendChatMessage } from "@/lib/api";
import { logger } from "@/lib/logger";

interface Message {
  role: "user" | "ai";
  content: string;
  adapter?: string;
}

interface ChatInterfaceProps {
  walletAddress: string;
}

export default function ChatInterface({ walletAddress }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [currentAdapter, setCurrentAdapter] = useState<string>("default_adapter");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    const userInput = input;
    setInput("");
    setIsLoading(true);

    try {
      let aiMessage: Message = { role: "ai", content: "", adapter: currentAdapter };
      setMessages((prev) => [...prev, aiMessage]);

      // Detect adapter from input
      const lowerInput = userInput.toLowerCase();
      let adapter = "default_adapter";
      if (
        lowerInput.includes("python") ||
        lowerInput.includes("code") ||
        lowerInput.includes("function") ||
        lowerInput.includes("bug") ||
        lowerInput.includes("error")
      ) {
        adapter = "coder_adapter";
      } else if (
        lowerInput.includes("law") ||
        lowerInput.includes("legal") ||
        lowerInput.includes("court") ||
        lowerInput.includes("rights")
      ) {
        adapter = "law_adapter";
      }
      setCurrentAdapter(adapter);

      await sendChatMessage(userInput, walletAddress, (chunk: string) => {
        aiMessage.content += chunk;
        setMessages((prev) => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1] = { ...aiMessage, adapter };
          return newMessages;
        });
      });
    } catch (error: any) {
      logger.error("Chat error:", error);
      const errorMessage: Message = {
        role: "ai",
        content: `Error: ${error.message || "Failed to get response"}`,
      };
      setMessages((prev) => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1] = errorMessage;
        return newMessages;
      });
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-900 overflow-hidden">
      {/* Mesaj Alanı - Scrollbar sadece burada */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4 min-h-0">
        {messages.length === 0 ? (
          /* Boş Durum */
          <div className="flex flex-col items-center justify-center h-full">
            <div className="text-center">
              <div className="text-6xl font-bold gradient-text mb-4">R3MES</div>
              <p className="text-slate-400 text-lg">Sisteme bağlı. Bir görev ver...</p>
            </div>
          </div>
        ) : (
          /* Mesaj Listesi */
          <>
            {messages.map((msg, idx) => (
              <div key={idx} className="text-sm">
                {msg.role === "user" ? (
                  <div className="flex justify-end mb-4">
                    <div className="bg-slate-800 rounded-lg px-4 py-3 max-w-2xl shadow-sm">
                      <span className="text-slate-200 whitespace-pre-wrap break-words">{msg.content}</span>
                    </div>
                  </div>
                ) : (
                  <div className="flex justify-start mb-4">
                    <div className="bg-slate-800 border-l-4 border-[#06b6d4] rounded-lg px-4 py-3 max-w-2xl shadow-sm">
                      {msg.adapter && (
                        <div className="text-xs text-slate-400 mb-2 font-medium">
                          Running '{msg.adapter}'...
                        </div>
                      )}
                      <div className="text-slate-200 whitespace-pre-wrap break-words">{msg.content}</div>
                      {msg.adapter && (
                        <div className="text-xs text-slate-500 mt-3 pt-2 border-t border-slate-700/50">
                          Model: BitNet-b1.58 | Router: Auto | Cost: 1 Credit
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start mb-4">
                <div className="bg-slate-800 border-l-4 border-[#06b6d4] rounded-lg px-4 py-3">
                  <div className="text-slate-400 text-sm flex items-center gap-2">
                    <span className="animate-pulse">●</span>
                    <span>Thinking...</span>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Alanı - En Alt (Sabit, her zaman görünür) */}
      <div className="border-t border-slate-700/50 p-4 bg-slate-800/30 flex-shrink-0">
        <div className="flex items-center gap-3 max-w-5xl mx-auto">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            disabled={isLoading}
            className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-[#06b6d4] focus:ring-2 focus:ring-[#06b6d4]/20 transition-all"
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="btn-primary px-6 py-3 disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
          >
            {isLoading ? "Sending..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
