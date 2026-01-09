import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

// Resolve docs directory path
// In Next.js, process.cwd() returns the project root (web-dashboard folder)
// docs folder is at the parent level: ../docs
// Use path.resolve to get absolute path
const ROOT = process.cwd();
const DOCS_DIR = path.resolve(ROOT, "..", "docs");

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ file: string }> | { file: string } }
) {
  // Only log in development mode
  if (process.env.NODE_ENV !== 'production') {
    console.log('[API Route] GET /api/docs/[file] called');
  }
  try {
    // Handle both sync and async params (Next.js 15+)
    const resolvedParams = typeof params === 'object' && params !== null && 'then' in params 
      ? await params 
      : params;
    const fileName = resolvedParams.file;
    if (process.env.NODE_ENV !== 'production') {
      console.log('[API Route] fileName:', fileName);
    }
    
    if (!fileName) {
      return NextResponse.json(
        { error: "File name is required" },
        { status: 400 }
      );
    }
    
    // Security: Only allow .md files and prevent directory traversal
    if (!fileName.endsWith(".md") || fileName.includes("..") || fileName.includes("/") || fileName.includes("\\")) {
      return NextResponse.json(
        { error: "Invalid file name" },
        { status: 400 }
      );
    }

    const filePath = path.join(DOCS_DIR, fileName);
    const resolvedDocsDir = path.resolve(DOCS_DIR);
    const resolvedFilePath = path.resolve(filePath);
    
    // Security check: ensure file is within docs directory (case-insensitive on Windows)
    const normalizedResolvedDocsDir = resolvedDocsDir.toLowerCase();
    const normalizedResolvedFilePath = resolvedFilePath.toLowerCase();
    if (!normalizedResolvedFilePath.startsWith(normalizedResolvedDocsDir)) {
      // Security errors should always be logged
      console.error(`Security: File path outside docs directory: ${resolvedFilePath} (DOCS_DIR: ${resolvedDocsDir})`);
      return NextResponse.json(
        { error: "Invalid file path" },
        { status: 403 }
      );
    }
    
    // Check if file exists
    if (!fs.existsSync(resolvedFilePath)) {
      // File not found errors should always be logged for debugging
      console.error(`File not found: ${resolvedFilePath} (DOCS_DIR: ${resolvedDocsDir}, cwd: ${process.cwd()}, ROOT: ${ROOT})`);
      return NextResponse.json(
        { error: `File not found: ${fileName}`, path: resolvedFilePath, docsDir: resolvedDocsDir },
        { status: 404 }
      );
    }

    const content = await fs.promises.readFile(resolvedFilePath, "utf-8");
    
    return new NextResponse(content, {
      headers: {
        "Content-Type": "text/markdown; charset=utf-8",
      },
    });
  } catch (error) {
    // Errors should always be logged (even in production)
    console.error("Error reading documentation file:", error);
    if (process.env.NODE_ENV !== 'production') {
      console.error("Error stack:", error instanceof Error ? error.stack : "No stack");
      console.error("DOCS_DIR:", DOCS_DIR);
      console.error("ROOT:", ROOT);
      console.error("process.cwd():", process.cwd());
    }
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    const errorStack = error instanceof Error ? error.stack : undefined;
    return NextResponse.json(
      { 
        error: "Internal server error", 
        details: errorMessage,
        stack: process.env.NODE_ENV === 'development' ? errorStack : undefined
      },
      { status: 500 }
    );
  }
}

