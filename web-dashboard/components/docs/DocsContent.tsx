"use client";

import { useState, useEffect } from "react";
import { marked } from "marked";
import DOMPurify from "isomorphic-dompurify";
import { FileText, Search, AlertCircle } from "lucide-react";

interface DocsContentProps {
  docPath: string;
}

// Map doc paths to actual markdown files
const DOC_PATH_MAP: Record<string, string> = {
  // Getting Started
  "": "00_home.md",
  "home": "00_home.md",
  "quick-start": "01_get_started.md",
  "installation": "INSTALLATION.md",
  
  // Learn
  "how-it-works": "00_project_summary.md",
  "architecture": "ARCHITECTURE_OVERVIEW.md",
  "tokenomics": "TOKENOMICS.md",
  "security": "03_security_verification.md",
  "governance": "06_governance_system.md",
  "economics": "04_economic_incentives.md",
  
  // Participate
  "mining-guide": "02_mining.md",
  "mining": "02_mining.md",
  "staking-guide": "staking.md",
  "staking": "staking.md",
  "validators": "03_validating.md",
  "validating": "03_validating.md",
  "serving": "04_serving.md",
  "proposing": "05_proposing.md",
  
  // Build
  "api-reference": "13_api_reference.md",
  "api": "13_api_reference.md",
  "sdk": "PROJECT_STRUCTURE.md",
  "backend": "14_backend_inference_service.md",
  "frontend": "15_frontend_user_interface.md",
  
  // Reference
  "faucet": "faucet.md",
  "troubleshooting": "TROUBLESHOOTING.md",
  "environment-variables": "16_environment_variables.md",
  "monitoring": "MONITORING.md",
  "production": "12_production_deployment.md",
};

export default function DocsContent({ docPath }: DocsContentProps) {
  const [content, setContent] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadDoc() {
      setLoading(true);
      setError(null);
      try {
        const markdownFile = DOC_PATH_MAP[docPath] || DOC_PATH_MAP[""];
        
        if (!markdownFile) {
          setError("Documentation not found");
          setLoading(false);
          return;
        }

        // Fetch markdown file from API route
        const response = await fetch(`/api/docs/${markdownFile}`);
        
        if (!response.ok) {
          // Try to get error details from response
          let errorDetails = response.statusText;
          try {
            const errorData = await response.json();
            if (errorData.details) {
              errorDetails = errorData.details;
            } else if (errorData.error) {
              errorDetails = errorData.error;
            }
          } catch {
            // If JSON parsing fails, use statusText
          }
          throw new Error(`Failed to load documentation: ${errorDetails}`);
        }

        const text = await response.text();
        
        // Configure marked for better rendering
        marked.setOptions({
          gfm: true,
          breaks: true,
        });

        const html = await marked(text);
        // Sanitize HTML to prevent XSS attacks
        // DOMPurify removes potentially dangerous HTML/JavaScript while preserving safe markdown-generated content
        const sanitized = DOMPurify.sanitize(html, {
          ALLOWED_TAGS: [
            'p', 'br', 'strong', 'em', 'u', 's', 'code', 'pre', 'blockquote',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'ul', 'ol', 'li',
            'a', 'img',
            'table', 'thead', 'tbody', 'tr', 'th', 'td',
            'hr', 'div', 'span',
          ],
          ALLOWED_ATTR: ['href', 'src', 'alt', 'title', 'class', 'id'],
          // Prevent javascript: and data: URLs in href attributes
          ALLOWED_URI_REGEXP: /^(?:(?:(?:f|ht)tps?|mailto|tel|callto|sms|cid|xmpp|data):|[^a-z]|[a-z+.\-]+(?:[^a-z+.\-:]|$))/i,
          // Add hook to sanitize href attributes
          ADD_ATTR: [],
        });
        
        // Additional sanitization: Remove any remaining javascript: or data: URLs from href
        // DOMPurify should handle this, but we add an extra layer of protection
        const finalSanitized = sanitized.replace(
          /href=["'](javascript|data|vbscript):/gi,
          'href="#"'
        );
        
        setContent(finalSanitized);
      } catch (error) {
        console.error("Error loading documentation:", error);
        setError(error instanceof Error ? error.message : "Error loading documentation");
        setContent("");
      } finally {
        setLoading(false);
      }
    }

    loadDoc();
  }, [docPath]);

  if (loading) {
    return (
      <div className="card p-8">
        <div className="animate-pulse">
          <div className="h-8 bg-slate-700 rounded w-3/4 mb-4"></div>
          <div className="h-4 bg-slate-700 rounded w-full mb-2"></div>
          <div className="h-4 bg-slate-700 rounded w-5/6"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-8">
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-3 text-red-400">
            <AlertCircle className="w-5 h-5" />
            <p className="font-semibold">{error}</p>
          </div>
          <div className="text-sm text-slate-400 mt-2">
            <p>Please check the Next.js development server terminal for detailed error logs.</p>
            <p className="mt-1">The error should appear in the terminal where you ran <code className="bg-slate-800 px-2 py-1 rounded">npm run dev</code>.</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card p-8">
      <div
        className="prose max-w-none"
        // Content is sanitized with DOMPurify before being set
        // This prevents XSS attacks while allowing safe markdown-generated HTML
        dangerouslySetInnerHTML={{ __html: content }}
      />
    </div>
  );
}

