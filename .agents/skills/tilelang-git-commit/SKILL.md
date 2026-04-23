---
name: tilelang-git-commit
description: Guidelines for creating Git commits in the tilelang repository. Handles staging, committing, and pushing changes while automatically excluding 3rdparty/ submodule modifications.
---

# Git Commit Guidelines

## Overview

When creating commits in the tilelang repository, special care must be taken to **exclude changes in `3rdparty/` submodules**. These submodules (`tvm/`, `cutlass/`, `composable_kernel/`) often show as modified due to their own internal git state, but their changes should **never** be included in root repository commits.

## Workflow

### 1. Check Status

```bash
git status
```

Look for:
- Files modified in the root directory (staged or unstaged)
- `3rdparty/` submodule changes — **these must be ignored**

### 2. Stage Root Changes Only

Stage only the files you want to commit, excluding `3rdparty/`:

```bash
# Stage specific files
git add <path/to/file1> <path/to/file2>

# Or stage all root changes but NOT 3rdparty:
git add -- . ':!3rdparty/'
```

> **Note**: `git add .` will **not** automatically include submodule changes unless `-u` or `--all` is used with submodule recursion. However, to be safe, always explicitly exclude `3rdparty/`.

### 3. Handle 3rdparty/ Changes

If `3rdparty/` shows as modified:

```bash
# Simply ignore it — do NOT add or commit it
# The submodule change is typically just internal state, not an intentional update

# Optionally, to discard the submodule change notice:
git submodule update --init --recursive 2>/dev/null || true
# Or just leave it as-is; it won't be committed
```

### 4. Commit with a Meaningful Message

```bash
# Conventional commit format:
# <type>[<scope>]: <description>
#
# Types:
#   feat     - New feature
#   fix      - Bug fix
#   refactor - Code restructuring
#   docs     - Documentation changes
#   test     - Test changes
#   chore    - Maintenance tasks
#   perf     - Performance improvements

git commit -m "<type>[<scope>]: <description>"
```

**Examples:**

```
git commit -m "[Docs] Add comprehensive T.vectorized analysis document"
git commit -m "[Build] Update CMake configuration for CUDA 12"
git commit -m "[Fix] Correct memory alignment in GEMM kernel"
```

For multi-line commit messages:

```bash
git commit -m "[<type>] <title>

- <bullet point 1>
- <bullet point 2>
- <bullet point 3>"
```

### 5. Push to Remote

```bash
# Push current branch to origin
git push origin <branch-name>
```

### Quick Script

For convenience, here's a one-liner to stage all root changes and commit:

```bash
git add -- . ':!3rdparty/' && git commit -m "<commit message>"
```

Or for a complete flow:

```bash
git add -- . ':!3rdparty/' && git commit -m "<message>" && git push origin $(git branch --show-current)
```

## Important Rules

| Rule | Description |
|------|-------------|
| **Never commit 3rdparty/** | Always exclude submodule changes |
| **Review before committing** | Use `git status` and `git diff --cached` to verify |
| **Meaningful messages** | Describe WHAT and WHY, not HOW |
| **One logical change per commit** | Keep commits focused and atomic |
