"use client";

import { useState, useRef, useEffect } from "react";
import { sendChatMessage } from "@/lib/api";
import { logger } from "@/lib/logger";
import { Send, Bot, User } from "lucide-react";
import { announceToScreenReader } from "@/utils/accessibility";

interface Message {
  role: "user" | "ai";
  content: string;
  adapter?: string;
  timestamp?: Date;
  id?: string;
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
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const [messageCount, setMessageCount] = useState(0);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Announce new messages to screen readers
  useEffect(() => {
    if (messages.length > messageCount) {
      const latestMessage = messages[messages.length - 1];
      if (latestMessage.role === 'ai') {
        // Announce AI response completion using utility function
        const truncatedContent = latestMessage.content.substring(0, 100);
        const suffix = latestMessage.content.length > 100 ? '...' : '';
        announceToScreenReader(`AI response received: ${truncatedContent}${suffix}`, 'polite');
      }
      setMessageCount(messages.length);
    }
  }, [messages, messageCount]);

  const generateMessageId = () => `msg-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const messageId = generateMessageId();
    const userMessage: Message = { 
      role: "user", 
      content: input.trim(), 
      timestamp: new Date(),
      id: messageId
    };
    setMessages((prev) => [...prev, userMessage]);
    const userInput = input.trim();
    setInput("");
    setIsLoading(true);

    // Announce message sending to screen readers using utility function
    announceToScreenReader('Message sent, waiting for AI response', 'polite');

    try {
      const aiMessageId = generateMessageId();
      let aiMessage: Message = { 
        role: "ai", 
        content: "", 
        adapter: currentAdapter, 
        timestamp: new Date(),
        id: aiMessageId
      };
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
        timestamp: new Date(),
        id: generateMessageId()
      };
      setMessages((prev) => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1] = errorMessage;
        return newMessages;
      });
      
      // Announce error to screen readers using utility function
      announceToScreenReader(
        `Error occurred: ${error.message || "Failed to get response"}`,
        'assertive',
        3000
      );
    } finally {
      setIsLoading(false);
      // Focus input after a small delay to ensure state has updated
      setTimeout(() => {
        inputRef.current?.focus();
      }, 0);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div 
      className="flex flex-col h-full bg-slate-900 overflow-hidden"
      role="main"
      aria-label="Chat interface"
    >
      {/* Chat Header */}
      <div 
        className="flex-shrink-0 border-b border-slate-700/50 p-4 bg-slate-800/30"
        role="banner"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Bot className="w-6 h-6 text-cyan-400" aria-hidden="true" />
            <div>
              <h1 className="text-lg font-semibold text-slate-200">R3MES AI Assistant</h1>
              <p className="text-sm text-slate-400">
                {currentAdapter === 'coder_adapter' ? 'Code Assistant Mode' : 
                 currentAdapter === 'law_adapter' ? 'Legal Assistant Mode' : 
                 'General Assistant Mode'}
              </p>
            </div>
          </div>
          <div 
            className="text-sm text-slate-400"
            role="status"
            aria-live="polite"
            aria-label={`${messages.length} messages in conversation`}
          >
            {messages.length} messages
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div 
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto p-6 space-y-4 min-h-0"
        role="log"
        aria-live="polite"
        aria-label="Chat messages"
        tabIndex={0}
      >
        {messages.length === 0 ? (
          /* Empty State */
          <div 
            className="flex flex-col items-center justify-center h-full"
            role="status"
            aria-label="Chat is empty, ready to start conversation"
          >
            <div className="text-center">
              <Bot className="w-16 h-16 text-cyan-400 mx-auto mb-4" aria-hidden="true" />
              <div className="text-6xl font-bold gradient-text mb-4" aria-hidden="true">R3MES</div>
              <p className="text-slate-400 text-lg">Sisteme bağlı. Bir görev ver...</p>
              <div className="mt-6 text-sm text-slate-500">
                <p>Tip: Try asking about code, legal questions, or general topics</p>
              </div>
            </div>
          </div>
        ) : (
          /* Message List */
          <>
            {messages.map((msg, idx) => (
              <div 
                key={msg.id || idx} 
                className="text-sm"
                role="article"
                aria-labelledby={`message-${idx}-label`}
              >
                {msg.role === "user" ? (
                  <div className="flex justify-end mb-4">
                    <div className="bg-slate-800 rounded-lg px-4 py-3 max-w-2xl shadow-sm">
                      <div className="flex items-center gap-2 mb-2">
                        <User className="w-4 h-4 text-slate-400" aria-hidden="true" />
                        <span 
                          id={`message-${idx}-label`}
                          className="text-xs text-slate-400 font-medium"
                        >
                          You
                        </span>
                        {msg.timestamp && (
                          <span className="text-xs text-slate-500 ml-auto">
                            {formatTime(msg.timestamp)}
                          </span>
                        )}
                      </div>
                      <div 
                        className="text-slate-200 whitespace-pre-wrap break-words"
                        aria-label={`Your message: ${msg.content}`}
                      >
                        {msg.content}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex justify-start mb-4">
                    <div className="bg-slate-800 border-l-4 border-[#06b6d4] rounded-lg px-4 py-3 max-w-2xl shadow-sm">
                      <div className="flex items-center gap-2 mb-2">
                        <Bot className="w-4 h-4 text-cyan-400" aria-hidden="true" />
                        <span 
                          id={`message-${idx}-label`}
                          className="text-xs text-slate-400 font-medium"
                        >
                          R3MES AI
                        </span>
                        {msg.timestamp && (
                          <span className="text-xs text-slate-500 ml-auto">
                            {formatTime(msg.timestamp)}
                          </span>
                        )}
                      </div>
                      {msg.adapter && (
                        <div 
                          className="text-xs text-slate-400 mb-2 font-medium"
                          aria-label={`Using ${msg.adapter} mode`}
                        >
                          Running '{msg.adapter}'...
                        </div>
                      )}
                      <div 
                        className="text-slate-200 whitespace-pre-wrap break-words"
                        aria-label={`AI response: ${msg.content}`}
                      >
                        {msg.content}
                      </div>
                      {msg.adapter && (
                        <div className="text-xs text-slate-500 mt-3 pt-2 border-t border-slate-700/50">
                          <span aria-label="Model information">
                            Model: BitNet-b1.58 | Router: Auto | Cost: 1 Credit
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start mb-4" role="status" aria-live="polite">
                <div className="bg-slate-800 border-l-4 border-[#06b6d4] rounded-lg px-4 py-3">
                  <div className="text-slate-400 text-sm flex items-center gap-2">
                    <span className="animate-pulse" aria-hidden="true">●</span>
                    <span>AI is thinking...</span>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
        <div ref={messagesEndRef} aria-hidden="true" />
      </div>

      {/* Input Area */}
      <div 
        className="border-t border-slate-700/50 p-4 bg-slate-800/30 flex-shrink-0"
        role="form"
        aria-label="Message input form"
      >
        <div className="flex items-center gap-3 max-w-5xl mx-auto">
          <div className="flex-1 relative">
            <label htmlFor="chat-input" className="sr-only">
              Type your message to R3MES AI
            </label>
            <input
              id="chat-input"
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message..."
              disabled={isLoading}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-[#06b6d4] focus:ring-2 focus:ring-[#06b6d4]/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              aria-describedby="chat-input-help"
              maxLength={2000}
            />
            <div id="chat-input-help" className="sr-only">
              Press Enter to send message, Shift+Enter for new line. Maximum 2000 characters.
            </div>
            {input.length > 1800 && (
              <div 
                className="absolute -top-8 right-2 text-xs text-slate-400"
                role="status"
                aria-live="polite"
              >
                {2000 - input.length} characters remaining
              </div>
            )}
          </div>
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="btn-primary px-6 py-3 disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap flex items-center gap-2 min-w-[100px] min-h-[44px]"
            aria-label={isLoading ? "Sending message" : "Send message"}
            type="submit"
          >
            {isLoading ? (
              <>
                <span className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full" aria-hidden="true" />
                <span>Sending...</span>
              </>
            ) : (
              <>
                <Send className="w-4 h-4" aria-hidden="true" />
                <span>Send</span>
              </>
            )}
          </button>
        </div>
        
        {/* Keyboard shortcuts help */}
        <div className="mt-2 text-xs text-slate-500 text-center">
          <span className="sr-only">Keyboard shortcuts: </span>
          Press <kbd className="px-1 py-0.5 bg-slate-700 rounded text-slate-300">Enter</kbd> to send, 
          <kbd className="px-1 py-0.5 bg-slate-700 rounded text-slate-300 ml-1">Shift+Enter</kbd> for new line
        </div>
      </div>
    </div>
  );
}
