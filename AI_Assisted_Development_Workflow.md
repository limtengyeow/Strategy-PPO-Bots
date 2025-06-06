
# 🚀 AI-Assisted Development Workflow (ChatGPT + Gemini + Human Execution)

This workflow enforces **strict change control** and **accountability** for AI-assisted development.

Roles:
- **ChatGPT** = *AI Coder* (proposes and writes code)
- **Gemini** = *AI Reviewer* (verifies code matches proposal exactly)
- **Human** = *Executor* (runs code, tests outputs, reports errors to ChatGPT)

No code is committed without Gemini’s validation and Human test confirmation.

---

## 🧭 Workflow Steps

### 1️⃣ ChatGPT: Propose Changes
- Describe proposed code modifications:
  - What will change
  - Why it’s needed
  - Expected impact

### 2️⃣ ChatGPT: Implement Code (Do NOT Commit Yet)
- Write only what’s proposed—no hidden edits or scope creep.
- Prepare `.md` archive entry with:
  - Version, Date
  - Git hash placeholder (`[pending]`)
  - Branch placeholder (`[pending]`)
  - Description
  - Full code snapshot

### 3️⃣ Gemini: Review
- Compare proposal and actual code:
  - Confirm **exact match**—no silent changes.
- Validate archive update accuracy.
- Provide feedback (✅ Approve or ❌ Request changes).

### 4️⃣ Human: Execute & Test Code
- Run code in the environment.
- Check outputs, logs, and performance.
- Report any errors, bugs, or unexpected behavior to ChatGPT.

### 5️⃣ ChatGPT: Iterate if Needed
- Fix only issues reported by Human or flagged by Gemini.
- Propose, implement, and update archive again.
- Repeat Gemini review and Human execution until all checks pass.

### 6️⃣ Gemini: Final Approval
- Ensure:
  - Code matches proposal.
  - Unit tests pass (`tests/` folder).
  - Data output verified (`data/` folder).
  - Archive updated (Git hash/branch/description).
- Mark Validation Checklist ✅ complete.

### 7️⃣ ChatGPT: Commit Code
- Only after Gemini's approval:
  ```bash
  git add <files>
  git commit -m "Implement [feature] as per approved proposal"
  ```
- Update `.md` archive with actual Git hash and branch.

---

## ✅ Validation Checklist (Per Version)

| Checklist Item                          | Status  |
|----------------------------------------|---------|
| Code matches ChatGPT’s proposed changes | ⬜ To Review |
| Unit tests pass (`tests/`)             | ⬜ To Review |
| Data output verified (`data/`)         | ⬜ To Review |
| Git hash/branch recorded correctly     | ⬜ To Review |
| Description accurate and complete      | ⬜ To Review |
| Reviewer (Gemini)                      | [Name]  |
| Date                                   | [Date]  |

---

## 🧭 Command Reference

- Get Git hash (after commit):
  ```bash
  git rev-parse HEAD
  ```

- Get current branch:
  ```bash
  git branch --show-current
  ```

---

## 🔄 Iterative Development Reminder

This is a **closed-loop process**:
- ChatGPT → Gemini → Human (Execution) → ChatGPT (Fixes) → Repeat until ✅.
- Commit only after final validation by Gemini and successful execution by Human.
