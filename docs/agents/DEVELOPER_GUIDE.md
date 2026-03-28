# CIAF Agentic Execution Boundaries - Developer Guide

## Overview

This guide provides developers with comprehensive information on integrating CIAF Agentic Execution Boundaries into AI agent systems.

## Table of Contents

1. [Architecture](#architecture)
2. [Core Concepts](#core-concepts)
3. [Integration Patterns](#integration-patterns)
4. [Best Practices](#best-practices)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                    ToolExecutor                         │
│  Orchestrates authorization and evidence recording      │
└───────────┬─────────────────────────┬───────────────────┘
            │                         │
            ↓                         ↓
┌───────────────────────┐   ┌────────────────────────────┐
│   PolicyEngine        │   │    EvidenceVault           │
│  - Evaluates requests │   │  - Records receipts        │
│  - Checks IAM/PAM     │   │  - Maintains chain         │
└───────┬───────────────┘   └────────────────────────────┘
        │
        ↓
┌───────────────────────┐   ┌────────────────────────────┐
│     IAMStore          │   │       PAMStore             │
│  - Identity mgmt      │   │  - Elevation grants        │
│  - Role/permissions   │   │  - Time-bound access       │
└───────────────────────┘   └────────────────────────────┘
```

### Data Flow

1. **Action Request** → Created by agent with justification
2. **Identity Resolution** → IAM store resolves identity and roles
3. **Policy Evaluation** → Engine checks RBAC + ABAC conditions
4. **Elevation Check** → If required, PAM validates active grant
5. **Tool Execution** → Executor invokes wrapped tool function
6. **Evidence Recording** → Vault generates cryptographic receipt
7. **Grant Update** → PAM tracks grant usage

[See full guide content above - truncated for brevity]

---

**Next Steps**: Check out the example scenarios in `examples/agents_scenarios/` for complete working implementations.
