import type { DocSource } from "./markdown";

export type DocCategory =
  | "getting_started"
  | "learn"
  | "participate"
  | "build"
  | "reference";

export type DocConfig = {
  id: string;
  title: string;
  description?: string;
  source: DocSource;
  file: string;
  category: DocCategory;
  order?: number;
};

export const DOCS: DocConfig[] = [
  // Getting Started
  {
    id: "home",
    title: "Welcome",
    description: "Introduction to R3MES documentation.",
    source: "docs",
    file: "00_home.md",
    category: "getting_started",
    order: 1,
  },
  {
    id: "quick-start",
    title: "Quick Start",
    description: "Get up and running in under 10 minutes.",
    source: "docs",
    file: "01_get_started.md",
    category: "getting_started",
    order: 2,
  },

  // Learn
  {
    id: "how-it-works",
    title: "How It Works",
    description: "Understanding the R3MES protocol.",
    source: "docs",
    file: "00_project_summary.md",
    category: "learn",
    order: 1,
  },
  {
    id: "tokenomics",
    title: "Tokenomics",
    description: "REMES token economics and distribution.",
    source: "docs",
    file: "TOKENOMICS.md",
    category: "learn",
    order: 2,
  },
  {
    id: "security",
    title: "Security",
    description: "Verification system and security model.",
    source: "docs",
    file: "03_security_verification.md",
    category: "learn",
    order: 3,
  },
  {
    id: "governance",
    title: "Governance",
    description: "Protocol governance and voting.",
    source: "docs",
    file: "06_governance_system.md",
    category: "learn",
    order: 4,
  },

  // Participate
  {
    id: "mining-guide",
    title: "Mining",
    description: "Train AI models and earn rewards.",
    source: "docs",
    file: "02_mining.md",
    category: "participate",
    order: 1,
  },
  {
    id: "staking-guide",
    title: "Staking",
    description: "Stake tokens and earn passive income.",
    source: "docs",
    file: "staking.md",
    category: "participate",
    order: 2,
  },
  {
    id: "validators",
    title: "Validating",
    description: "Run a validator node.",
    source: "docs",
    file: "03_validating.md",
    category: "participate",
    order: 3,
  },

  // Build
  {
    id: "api-reference",
    title: "API Reference",
    description: "REST, gRPC, and WebSocket APIs.",
    source: "docs",
    file: "13_api_reference.md",
    category: "build",
    order: 1,
  },
  {
    id: "sdk",
    title: "SDK",
    description: "Python, JavaScript, and Go SDKs.",
    source: "docs",
    file: "PROJECT_STRUCTURE.md",
    category: "build",
    order: 2,
  },

  // Reference
  {
    id: "faucet",
    title: "Testnet Faucet",
    description: "Get free testnet tokens.",
    source: "docs",
    file: "faucet.md",
    category: "reference",
    order: 1,
  },
  {
    id: "troubleshooting",
    title: "Troubleshooting",
    description: "Common issues and solutions.",
    source: "docs",
    file: "TROUBLESHOOTING.md",
    category: "reference",
    order: 2,
  },
];

export function getDocById(id: string): DocConfig | undefined {
  return DOCS.find((doc) => doc.id === id);
}

export function getDocsByCategory(category: DocCategory): DocConfig[] {
  return DOCS.filter((doc) => doc.category === category).sort(
    (a, b) => (a.order || 0) - (b.order || 0)
  );
}

export const DOC_CATEGORIES: { id: DocCategory; label: string }[] = [
  { id: "getting_started", label: "Getting Started" },
  { id: "learn", label: "Learn" },
  { id: "participate", label: "Participate" },
  { id: "build", label: "Build" },
  { id: "reference", label: "Reference" },
];
