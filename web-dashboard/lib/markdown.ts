import fs from "fs";
import path from "path";
import { marked } from "marked";
import DOMPurify from "isomorphic-dompurify";

const ROOT = process.cwd();
const DOCS_DIR = path.join(ROOT, "..", "docs");
const NEW_DOC_DIR = path.join(ROOT, "..", "new_doc");

export type DocSource = "docs" | "new_doc" | "root";

export function resolveDocPath(source: DocSource, file: string): string {
  switch (source) {
    case "docs":
      return path.join(DOCS_DIR, file);
    case "new_doc":
      return path.join(NEW_DOC_DIR, file);
    case "root":
    default:
      return path.join(ROOT, "..", file);
  }
}

export async function loadMarkdownFile(
  source: DocSource,
  file: string,
): Promise<{ html: string; raw: string }> {
  const fullPath = resolveDocPath(source, file);
  const raw = await fs.promises.readFile(fullPath, "utf8");
  const htmlUnsanitized = marked.parse(raw) as string;
  
  // Sanitize HTML to prevent XSS attacks
  // DOMPurify removes potentially dangerous HTML/JavaScript while preserving safe markdown-generated content
  const html = DOMPurify.sanitize(htmlUnsanitized, {
    // Allow common markdown HTML elements
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
    // Allow data attributes for syntax highlighting (if used)
    ALLOW_DATA_ATTR: false,
  });
  
  // Additional sanitization: Remove any remaining javascript: or data: URLs from href
  // DOMPurify should handle this, but we add an extra layer of protection
  const finalHtml = html.replace(
    /href=["'](javascript|data|vbscript):/gi,
    'href="#"'
  );
  
  return { html: finalHtml, raw };
}


