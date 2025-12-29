"use client";

import { useState, useEffect, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import "./NotificationCenter.css";

interface Notification {
  id: string;
  title: string;
  message: string;
  priority: "low" | "medium" | "high" | "critical";
  type: "mining" | "system" | "economic" | "governance";
  read: boolean;
  created_at: string;
  metadata?: Record<string, any>;
}

export default function NotificationCenter() {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const queryClient = useQueryClient();

  const { data: notifications = [], isLoading } = useQuery<Notification[]>({
    queryKey: ["notifications"],
    queryFn: async () => {
      const response = await fetch("/api/notifications");
      if (!response.ok) {
        // Return empty array if endpoint not available
        return [];
      }
      return response.json();
    },
    refetchInterval: 30000,
    retry: false,
  });

  const markAsReadMutation = useMutation({
    mutationFn: async (notificationId: string) => {
      await fetch(`/api/notifications/${notificationId}/read`, { method: "POST" });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["notifications"] });
    },
  });

  const markAllAsReadMutation = useMutation({
    mutationFn: async () => {
      await fetch("/api/notifications/read-all", { method: "POST" });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["notifications"] });
    },
  });

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const unreadCount = notifications.filter(n => !n.read).length;

  const getPriorityIcon = (priority: string) => {
    switch (priority) {
      case "critical": return "🔴";
      case "high": return "🟠";
      case "medium": return "🟡";
      default: return "🔵";
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case "mining": return "⛏️";
      case "system": return "⚙️";
      case "economic": return "💰";
      case "governance": return "🗳️";
      default: return "📢";
    }
  };

  const formatTime = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return "Just now";
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="notification-center" ref={dropdownRef}>
      <button 
        className="notification-bell"
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Notifications"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
          <path d="M13.73 21a2 2 0 0 1-3.46 0" />
        </svg>
        {unreadCount > 0 && (
          <span className="notification-badge">
            {unreadCount > 99 ? "99+" : unreadCount}
          </span>
        )}
      </button>

      {isOpen && (
        <div className="notification-dropdown">
          <div className="notification-header">
            <h3>Notifications</h3>
            {unreadCount > 0 && (
              <button 
                onClick={() => markAllAsReadMutation.mutate()}
                className="mark-all-read"
                disabled={markAllAsReadMutation.isPending}
              >
                Mark all as read
              </button>
            )}
          </div>

          <div className="notification-list">
            {isLoading ? (
              <div className="notification-loading">Loading...</div>
            ) : notifications.length === 0 ? (
              <div className="no-notifications">
                <span className="no-notifications-icon">🔔</span>
                <p>No notifications yet</p>
              </div>
            ) : (
              notifications.slice(0, 20).map(notification => (
                <div 
                  key={notification.id}
                  className={`notification-item priority-${notification.priority} ${notification.read ? 'read' : 'unread'}`}
                  onClick={() => !notification.read && markAsReadMutation.mutate(notification.id)}
                >
                  <div className="notification-icon">
                    <span className="type-icon">{getTypeIcon(notification.type)}</span>
                    <span className="priority-indicator">{getPriorityIcon(notification.priority)}</span>
                  </div>
                  <div className="notification-content">
                    <div className="notification-title">{notification.title}</div>
                    <div className="notification-message">{notification.message}</div>
                    <div className="notification-meta">
                      <span className="notification-time">{formatTime(notification.created_at)}</span>
                      <span className="notification-type">{notification.type}</span>
                    </div>
                  </div>
                  {!notification.read && (
                    <div className="unread-indicator" />
                  )}
                </div>
              ))
            )}
          </div>

          {notifications.length > 0 && (
            <div className="notification-footer">
              <a href="/notifications" className="view-all-link">
                View all notifications
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
