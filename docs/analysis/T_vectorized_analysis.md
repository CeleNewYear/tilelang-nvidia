# T.vectorized 深度分析文档

> 本文档基于 TileLang 源码对 `T.vectorized` 进行全面分析，涵盖其作用、编译管线处理流程、各优化场景的优化前后对比以及实用建议。

---

## 目录

1. [T.vectorized 是什么](#1-tvectorized-是什么)
2. [编译管线全流程](#2-编译管线全流程)
3. [核心机制：VectorizePlanner 自动规划向量宽度](#3-核心机制vectorizeplanner-自动规划向量宽度)
4. [核心机制：TLVectorizer 向量化重写](#4-核心机制tlvectorizer-向量化重写)
5. [场景一：基本向量化加载/存储](#5-场景一基本向量化加载存储)
6. [场景二：256-bit 宽向量化扩展](#6-场景二256-bit-宽向量化扩展)
7. [场景三：DecoupleTypeCast 混合精度解耦](#7-场景三decoupletypecast-混合精度解耦)
8. [场景四：向量化 + 越界安全检测 (LegalizeSafeMemoryAccess)](#8-场景四向量化--越界安全检测-legalizesafememoryaccess)
9. [场景五：cp.async 异步拷贝向量化展开](#9-场景五-cpasync-异步拷贝向量化展开)
10. [场景六：原子操作向量化合并](#10-场景六原子操作向量化合并)
11. [场景七：标量化回退 (无法向量化的情形)](#11-场景七标量化回退-无法向量化的情形)
12. [场景八：嵌套循环向量化 (外循环 + 内层向量化)](#12-场景八嵌套循环向量化-外循环--内层向量化)
13. [场景九：T.vectorized 与 T.Parallel 配合](#13-场景九-tvectorized-与-tparallel-配合)
14. [场景十：T.vectorized 与 swizzle 布局的协同](#14-场景十-tvectorized-与-swizzle-布局的协同)
15. [优化建议与最佳实践](#15-优化建议与最佳实践)
16. [附录：关键配置项](#16-附录关键配置项)

---

## 1. T.vectorized 是什么

`T.vectorized` 是 TileLang 中的一种**循环标记**，用于声明循环体应被编译器**向量化**——将多个标量操作合并为单个 SIMD 向量操作。

### 定义位置

| 层级 | 文件 |
|------|------|
| Python 前端导出 | `tilelang/language/__init__.py` 导出 `vectorized` |
| Python 实现 | `tilelang/language/loop.py` → 调用 `tb_tir.vectorized()` |
| TIR 中转 | `tilelang/language/tir/ir.py` → 调用 TVM 内置 `_ir.vectorized()` |
| C++ IR 表示 | `ForKind::kVectorized` (TVM TIR 层) |

### 基本语法

```python
for vec in T.vectorized(N):
    # 循环体中对连续内存的访问
    A[base + vec] = B[base + vec] + C[base + vec]
```

---

## 2. 编译管线全流程

`T.vectorized` 经历以下编译阶段：

```
┌─────────────────────────────────────────────────────┐
│                 用户代码                             │
│   for vec in T.vectorized(16):                      │
│       A[vec] = B[vec] + 1.0                         │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  LowerAndLegalize 阶段                               │
│                                                      │
│  ① DecoupleTypeCast                                  │
│     └ 拆分混合精度向量化循环                           │
│                                                      │
│  ② LegalizeVectorizedLoop                            │
│     └ kVectorized → kSerial + 自动规划向量宽度         │
│                                                      │
│  ③ LegalizeSafeMemoryAccess                          │
│     └ 插入/省略越界安全守卫                            │
│                                                      │
│  ④ Simplify                                          │
│     └ 简化表达式                                      │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  OptimizeForTarget 阶段                               │
│                                                      │
│  ⑤ FlattenBuffer                                     │
│     └ 展平多维buffer为一维                              │
│                                                      │
│  ⑥ VectorizeLoop ← 真正的向量化展开！                  │
│     └ TLVectorizer 将串行循环变为向量化循环              │
│                                                      │
│  ⑦ StorageRewrite                                    │
│     └ 存储重写                                         │
│                                                      │
│  ⑧ LowerLDGSTG                                       │
│     └ Ramp → ldg128/stg128 指令                       │
│                                                      │
│  ⑨ 后续代码生成                                       │
│     └ PTX/SASS 生成                                   │
└─────────────────────────────────────────────────────┘
```

### 关键步骤详解

#### 步骤② LegalizeVectorizedLoop

```cpp
// src/transform/legalize_vectorized_loop.cc
// 核心逻辑：将 kVectorized 转为 kSerial 后调用 VectorizeLoop 自动展开
Stmt VisitStmt_(const ForNode *op) final {
    For for_node = Downcast<For>(IRMutatorWithAnalyzer::VisitStmt_(op));
    if (for_node->kind != ForKind::kVectorized) {
        return IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    // 关键：改为串行循环，让 VectorizeLoop pass 后续处理
    for_node.CopyOnWrite()->kind = ForKind::kSerial;
    // 立即调用向量化规划
    return VectorizeLoop(for_node, analyzer_);
}
```

#### 步骤⑥ VectorizeLoop

```cpp
// src/transform/vectorize_loop.cc
// 遍历所有 For 节点，对 kVectorized 的循环执行向量化
Stmt VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kVectorized) {
        return TLVectorizer::Vectorize(op->loop_var, op->extent, op->body);
    }
    return StmtMutator::VisitStmt_(op);
}
```

---

## 3. 核心机制：VectorizePlanner 自动规划向量宽度

`VectorizePlanner`（`src/transform/loop_vectorize.cc`）是**自动决定最佳向量宽度**的核心。

### 决策流程

```
输入：待向量化的循环
                │
                ▼
        收集所有 Buffer 访问信息
                │
                ▼
        分类 buffer 类型
        ┌────────────┬────────────┬──────────────┐
        │local/frag  │ global     │ shared       │
        │(寄存器级)   │ (全局内存)  │ (共享内存)    │
        └────────────┴────────────┴──────────────┘
                │
                ▼
        应用向量化策略
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
 有SeqStmt   有全局/共享   仅local/frag
 (多条语句)   (且单条语句)   (纯寄存器操作)
    │           │           │
    ▼           ▼           ▼
 取全部buffer  忽略local    取local和
 的GCD(保守)   只取global/  call_node的
               shared的GCD  GCD
                │
                ▼
         与循环范围取 GCD
                │
                ▼
         最终 vector_size
```

### 影响向量宽度的因素

| 因素 | 说明 | 示例 |
|------|------|------|
| **数据类型位宽** | 决定最大向量宽度 | float32 → 128/32=4 个元素 |
| **向量化模式** | 128-bit 或 256-bit | 纯全局内存且支持时可用 256-bit |
| **循环范围** | vector_size 必须整除 extent | extent=32, vector_size 只能是 16,8,4,2,1 |
| **内存对齐** | 基地址偏移必须能被向量宽度整除 | `A[base + vec]`, base % vector_size == 0 |
| **连续索引** | 访问模式必须是连续 Ramp | `A[vec*2]` 不连续 → 无法向量化 |
| **Cast 节点** | 混合精度限制向量宽度 | fp32+fp8 → GCD(4,16)=4 |
| **循环体结构** | 多条语句时取保守 GCD | SeqStmt → 所有 buffer 的最小公约束 |

### 关键源码分析

```cpp
// 从 src/transform/loop_vectorize.cc 中提取的核心策略
if (has_seq_stmt) {
    // 策略1：保守策略 - 取所有 buffer 的 GCD
    vector_size_ = GCD(GCD(local_fragment_min, memory_min), call_node_min);
} else if (has_global_or_shared_buffer) {
    // 策略2：忽略 local/fragment 约束，只考虑 memory
    // 因为 local/fragment 是寄存器级的，没有内存对齐约束
    vector_size_ = GCD(memory_min, non_cast_call_node_min);
} else {
    // 策略3：只有 local/fragment，取 GCD
    vector_size_ = GCD(local_fragment_min, call_node_min);
}
```

---

## 4. 核心机制：TLVectorizer 向量化重写

`TLVectorizer`（`src/transform/vectorize_loop.cc`）是实际的向量化重写引擎。

### 工作原理

将循环变量替换为 `Ramp` 表达式：

```cpp
// 向量化前
for (int i = 0; i < 16; i++) {
    A[i] = B[i] + 1;
}

// 向量化后 - TLVectorizer 将 i 替换为 Ramp(0, 1, 16)
// 相当于:
//   A[ramp(0,1,16)] = B[ramp(0,1,16)] + broadcast(1, 16)
// 最终生成:
//   LDG.128 加载 B[0..3], B[4..7], B[8..11], B[12..15]
//   4次向量加载完成
```

### 表达式向量化规则

| 表达式 | 向量化方式 |
|--------|-----------|
| `Var` (循环变量) | → `Ramp(0, 1, vector_size)` |
| `Add(a, b)` | 若 `a=Ramp, b=标量` → `Ramp(a.base+b, a.stride, a.lanes)` |
| `Mul(a, b)` | 若 `a=Ramp, b>0标量` → `Ramp(a.base*b, a.stride*b, a.lanes)` |
| `BufferLoad(buf, [i])` | i 变为 Ramp → 加载 `buf.dtype × lanes` 宽的数据 |
| `Cast` | 保持 lanes 数，改变 dtype |
| `Broadcast(a, lanes)` | 若 a 已向量化 → 标量化回退 |
| `if_then_else(cond, t, f)` | 若 cond 是向量 → 标量化回退 |

---

## 5. 场景一：基本向量化加载/存储

### 场景描述

最简单的向量化场景：连续内存访问，所有条件都满足。

### 代码示例

```python
# ===== 优化前 (用户代码) =====
@T.prim_func
def main(A: T.Tensor((128,), float32), B: T.Tensor((128,), float32), 
         C: T.Tensor((128,), float32)):
    with T.Kernel(1, 1, threads=128) as (bx, by):
        tid = T.get_thread_binding()
        for vec in T.vectorized(4):          # ← 每个线程处理4个元素
            idx = tid * 4 + vec
            C[idx] = A[idx] + B[idx]

# ===== 编译中间表示 (VectorizePlanner 规划后) =====
# vector_size = 4 (128-bit / 32-bit = 4)
# 循环范围 4 = vector_size → 直接标记为 kVectorized

# ===== 向量化后 (TLVectorizer 展开后) =====
# Tl + SIMT 模型下:
# 每个线程:
#   C[ramp(tid*4, 1, 4)] = A[ramp(tid*4, 1, 4)] + B[ramp(tid*4, 1, 4)]

# ===== 最终生成 PTX 伪代码 =====
# LDG.128  [R0], [A + tid*4]     // 一次加载 16 字节 (4×float32)
# LDG.128  [R1], [B + tid*4]     // 一次加载 16 字节
# FADD.32  R2, R0, R1             // 4 lane 向量加法
# STG.128  [C + tid*4], R2        // 一次存储 16 字节
```

### 内存访问对比

```
向量化前 (标量):
  LDG.32  R0, [A+0]    ← 4 次加载
  LDG.32  R1, [A+4]
  LDG.32  R2, [A+8]
  LDG.32  R3, [A+12]
  FADD.32 R4, R0, R1   ← 4 次加法
  FADD.32 R5, R2, R3
  FADD.32 R6, ...
  FADD.32 R7, ...
  STG.32  [C+0], R4    ← 4 次存储
  STG.32  [C+4], R5
  STG.32  [C+8], R6
  STG.32  [C+12], R7

向量化后 (128-bit):
  LDG.128 R0, [A+0]    ← 1 次加载 4 个元素
  LDG.128 R1, [B+0]    ← 1 次加载 4 个元素
  FADD.32 R2, R0, R1   ← 1 次向量加法 (4 lane)
  STG.128 [C+0], R2    ← 1 次存储 4 个元素

收益: 指令数减少 75%，内存事务数减少 75%
```

### 优化效果

| 指标 | 标量版本 | 向量化版本 (128-bit) | 提升 |
|------|---------|-------------------|------|
| 加载指令数 | 4 | 1 | 4× |
| 存储指令数 | 4 | 1 | 4× |
| 内存事务数 | 4 × 32bit | 1 × 128bit | 4× |
| 计算指令数 | 4 | 1 (向量) | 4× |

---

## 6. 场景二：256-bit 宽向量化扩展

### 场景描述

当只有全局内存访问且目标 GPU 支持时，可以启用 256-bit 向量化（8×float32 或 16×float16）。

### 触发条件

```cpp
// src/transform/loop_vectorize.cc
if (TargetSupportVectorize256(Target::Current(false)) &&
    !disable_vectorize_256 &&
    VectorizeFindMemoryAccess::MaySupportVectorize256(node)) {
    // ↓ 纯全局内存访问，无共享内存
    vector_load_bits_max_ = 256;  // 启用 256-bit
} else {
    vector_load_bits_max_ = 128;  // 默认 128-bit
}
```

条件：
1. **目标 GPU 支持**（SM 版本足够高）
2. **纯全局内存访问**（无 shared 或 local buffer 参与）
3. **未被配置禁用**

### 代码示例

```python
# ===== 优化前 =====
@T.prim_func
def main(A: T.Tensor((256,), float16), B: T.Tensor((256,), float16),
         C: T.Tensor((256,), float16)):
    with T.Kernel(1, 1, threads=32) as (bx, by):
        tid = T.get_thread_binding()
        # 只有全局内存访问 (A, B, C 都是 global)
        for vec in T.vectorized(8):    # ← 8×float16 = 128-bit 用户指定
            idx = tid * 8 + vec
            C[idx] = A[idx] + B[idx]

# ===== VectorizePlanner 决策 =====
# ✓ 只有全局内存访问 (A, B, C 都是 global)
# ✓ 目标 GPU 支持 256-bit
# → 自动提升到 256-bit
# vector_size = 256/16 = 16

# ===== 自动展开后 =====
# 外循环
for outer in T.serial(2):    # 原 8 个元素 → 256-bit 一次处理 16 个，但用户只想 8 个
    for vec in T.vectorized(8):
        idx = tid * 16 + outer * 8 + vec
        C[idx] = A[idx] + B[idx]

# 更准确来说，如果 extent=16，vectorize 提升到 16：
for vec in T.vectorized(16):
    C[tid*16 + vec] = A[tid*16 + vec] + B[tid*16 + vec]
# 生成 LDG.256 / STG.256 指令
```

### 限制条件

```python
# 以下情况即使满足条件也无法启用 256-bit 向量化:
# ❌ 有共享内存访问
@T.prim_func
def main(A: T.Tensor((128,), float32), B: T.Tensor((128,), float32)):
    with T.Kernel(1, 1, threads=128) as (bx, by):
        tid = T.get_thread_binding()
        A_shared = T.alloc_shared((128,), float32)  # ← 共享内存!
        for vec in T.vectorized(4):
            A_shared[vec] = A[tid*4 + vec]          # 无法启用 256-bit
            # 因为 shared 参与了访问
```

---

## 7. 场景三：DecoupleTypeCast 混合精度解耦

### 场景描述

当 `T.vectorized` 循环中包含 `T.cast` 时，不同数据类型有不同的最大向量宽度，取 GCD 会限制性能。`DecoupleTypeCast` 通过拆分循环来解除这种约束。

### 源码位置

```python
# tilelang/transform/decouple_type_cast.py
# tilelang/engine/phase.py 中:
# mod = tilelang.transform.DecoupleTypeCast()(mod)
# 必须在 VectorizeLoop 和 StorageRewrite 之前执行
```

### 问题示例

```python
# ===== 优化前：混合精度约束 =====
@T.prim_func
def main(
    a_frag: T.Tensor((64,), float32),  # fragment (寄存器)
    b: T.Tensor((64,), "float4_e2m1fn")  # global (fp8)
):
    with T.Kernel(1, 1, threads=4) as (bx, by):
        tid = T.get_thread_binding()
        for vec in T.vectorized(16):     # ← 用户想一次处理 16 个
            # 同时包含 fp32 和 fp8 操作 
            b[vec] = T.cast(a_frag[vec], "float4_e2m1fn")
            #        ^^^^^           ^^^^^^^^
            #       fp32 操作        目标 fp8

# 向量化分析:
# - a_frag[vec]: float32 → 支持 128/32 = 4 lane 向量化
# - b[vec]: float8 → 支持 128/8 = 16 lane 向量化
# - cast: 需要同时处理 fp32 和 fp8 → GCD(4, 16) = 4
# 限制: vector_size 只能 = 4（除非解耦）
```

### 优化后：DecoupleTypeCast 解耦

```python
# ===== 优化后 =====
# DecoupleTypeCast 自动插入中间缓冲区，将计算和拷贝分离：

# 步骤1：插入 cast 临时缓冲区
cast_buf = T.alloc_buffer((64,), "float4_e2m1fn", scope="local")

# 步骤2：分离为两个循环，各用最佳向量宽度
# 计算循环（用 float32 宽度 vec=4）
for vec in T.vectorized(16):
    cast_buf[vec] = T.cast(a_frag[vec], "float4_e2m1fn")
    # ^^^^ fp32 fragment, vec 实际按 4 宽度处理

# 拷贝循环（用 float8 宽度 vec=16）
for vec_copy in T.vectorized(16):
    b[vec_copy] = cast_buf[vec_copy]
    # 批量连续拷贝到 global memory，用 16 宽度
```

### 另一个方向：从内存加载 + cast

```python
# ===== 优化前 =====
@T.prim_func
def main(
    b: T.Tensor((64,), "float4_e2m1fn"),
    a_frag: T.Tensor((64,), float32),
):
    with T.Kernel(1, 1, threads=4) as (bx, by):
        tid = T.get_thread_binding()
        for vec in T.vectorized(16):
            a_frag[vec] = T.cast(b[vec], "float32")

# ===== 优化后 =====
# 步骤1：插入 cast 临时缓冲区
cast_buf = T.alloc_buffer((64,), "float4_e2m1fn", scope="local")

# 步骤2：分离
# 拷贝循环（用 float8 宽度 vec=16）
for vec_copy in T.vectorized(16):
    cast_buf[vec_copy] = b[vec_copy]

# 计算循环（用 float32 宽度 vec=4）
for vec in T.vectorized(16):
    a_frag[vec] = T.cast(cast_buf[vec], "float32")
```

### 内存访问模式对比

```
优化前 (GCD=4):
  LDG.32  [R0], [b+0]    ← fp8 但只加载 4 个(32bit)
  LDG.32  [R1], [b+4]
  LDG.32  [R2], [b+8]
  LDG.32  [R3], [b+12]   ← 共4次，实际上可以一次 LDG.128 加载 16 个 fp8
  CVT.fp32.fp8 ...
  ...

优化后 (解耦, vec=16 for copy):
  LDG.128 [R0], [b+0]    ← 一次加载 16 个 fp8 (128bit)
  LDG.128 [R1], [b+16]   ← 二次加载完成
  ...
  CVT.fp32.fp8 ...       ← 计算时用 fp32 宽度
  ...
```

### 优化效果

| 指标 | 优化前 (GCD=4) | 优化后 (解耦) | 提升 |
|------|--------------|-------------|------|
| 内存加载次数 (fp8) | 16 次 32-bit | 4 次 128-bit | 4× |
| 计算宽度 (fp32) | 4 lane | 16 lane | 4× |
| 总带宽利用率 | 25% | 100% | 4× |

### 注意事项

> **Note**: `DecoupleTypeCast` 必须在 `VectorizeLoop` 和 `StorageRewrite` 之前执行，因为此时 IR 仍使用 `BufferLoad/BufferStore`（而非 `tvm_access_ptr`）。

---

## 8. 场景四：向量化 + 越界安全检测 (LegalizeSafeMemoryAccess)

### 场景描述

当向量化访问可能越界时（如最后一个 tile 的残余元素），`LegalizeSafeMemoryAccess` 自动插入边界守卫，保证不会访问非法内存。

### 源码位置

```cpp
// src/transform/legalize_safe_memory_access.cc
tvm::transform::Pass LegalizeSafeMemoryAccess()
```

### 安全的向量化访问（编译器可证明在边界内）

```python
# ===== 优化前 =====
@T.prim_func
def main(A: T.Tensor((128, 64), float32)):
    with T.Kernel(1, 1, threads=64) as (bx, by):
        tid = T.get_thread_binding()
        for vec in T.vectorized(4):
            # tid ∈ [0, 64), A.shape[0]=128 → tid < 128 恒成立
            # j ∈ [0, 64),  A.shape[1]=64  → j < 64 恒成立
            A[tid, vec] = 1.0

# ===== 编译后：无插入 =====
# 编译器通过区间分析:
#   CanProve(tid < 128) = True
#   CanProve(vec < 64)  = True (因为 extent=4 < 64)
# → 不插入任何边界检查代码
# → 直接生成 STG.128
```

### 不安全的向量化访问（编译器无法证明在边界内）

```python
# ===== 优化前 =====
@T.prim_func
def main(A: T.Tensor((64, 64), float32)):
    with T.Kernel(1, 1, threads=64) as (bx, by):
        tid = T.get_thread_binding()
        for vec in T.vectorized(4):
            # tid+2 ∈ [2, 65), A.shape[0]=64 → 可能越界!
            A[tid + 2, vec] = 1.0

# ===== 编译后：自动插入守卫 =====
# 编译器分析:
#   CanProve(tid+2 < 64) = False (tid=63 时越界)
# → BufferStore → 用 IfThenElse 包裹

# 生成的等价代码:
for vec in T.vectorized(4):
    if tid + 2 < 64:          # ← 自动插入的行边界检查
        A[tid + 2, vec] = 1.0  # 安全写入
    # ← 越界时自动跳过写入
```

### 向量化读 + 越界 → 安全值

```python
# ===== 优化前 =====
@T.prim_func
def main(A: T.Tensor((64, 64), float32)):
    with T.Kernel(1, 1, threads=64) as (bx, by):
        tid = T.get_thread_binding()
        B_shared = T.alloc_shared((64, 64), float32)
        for vec in T.vectorized(4):
            # 读可能越界
            B_shared[tid, vec] = A[tid + 2, vec]

# ===== 编译后：自动插入安全值 =====
for vec in T.vectorized(4):
    # BufferLoad 越界 → 返回 0.0 (安全值)
    B_shared[tid, vec] = T.if_then_else(
        tid + 2 < 64,                # 条件
        A[tid + 2, vec],              # 安全加载
        T.float32(0)                  # 越界时返回 0
    )
```

### 向量化写入 + 用户已有 if 处理

```python
# ===== 用户代码：已自行处理边界 =====
@T.prim_func
def main(A: T.Tensor((70,), float32)):
    with T.Kernel(1, 1, threads=32) as (bx, by):
        tid = T.get_thread_binding()
        for vec in T.vectorized(4):
            idx = tid * 4 + vec
            # 用户已自行处理
            if idx < 70:
                A[idx] = 1.0

# ===== 编译后：跳过自动检测 =====
# 检测到 store 的 value 已经是 IfThenElse
# → 认为用户已自行处理
# → LOG(WARNING) 但跳过自动守卫插入
```

### 自定义安全值

```python
# ===== 用户代码：自定义安全值 =====
@T.prim_func
def main(A: T.Tensor((16,), float16)):
    with T.block("root"):
        # 自定义越界时的安全值 = 3.0
        T.block_attr({"safe_value_map": {A.data: T.float16(3)}})
        A_shared = T.alloc_buffer((16,), float16, scope="shared")
        for i in T.serial(4):
            for vec in T.vectorized(4):
                idx = i * 4 + vec
                T.ptx_cp_async(
                    T.access_ptr(A_shared[idx], "w", 4),
                    T.access_ptr(A[idx + 10], "r", 4),
                    4
                )

# ===== 编译后 =====
# 安全值 ≠ 0 → 不能用 PTX 条件拷贝
# → 改为 if-else 结构:
for i in T.serial(4):
    for vec in T.vectorized(4):
        idx = i * 4 + vec
        if idx + 10 < 16:
            T.ptx_cp_async(..., A[idx+10], ...)  # 正常拷贝
        else:
            A_shared[idx] = T.float16(3)  # 越界 → 写入安全值 3.0
```

---

## 9. 场景五：cp.async 异步拷贝向量化展开

### 场景描述

PTX `cp.async` 指令用于从全局内存异步拷贝到共享内存。向量化时可以将其合并为更宽的传输。

### 代码示例

```python
# ===== 优化前 =====
@T.prim_func
def main(A: T.Tensor((64,), float16), A_shared: T.Buffer((64,), float16, scope="shared")):
    with T.Kernel(1, 1, threads=4) as (bx, by):
        tid = T.get_thread_binding()
        for vec in T.vectorized(4):     # ← vec=4, 每个 cp.async 传 4×8bit=32bit
            idx = tid * 4 + vec
            T.ptx_cp_async(
                T.access_ptr(A_shared[idx], "w", 4),
                T.access_ptr(A[idx], "r", 4),
                4                        # 传输 4 个元素
            )

# ===== TLVectorizer 向量化展开 =====
# 每个 cp.async 的传输宽度乘以 vector_size
# bits_per_call = 4 * 8 = 32bit (每个元素 float16=16bit, 4个 = 64bit... 等等)
# 实际: ptx_cp_async 的参数 count 是字节数
# 4 个 float16 = 8 字节
# vector_size=4 → total = 4 * 4 * 16bit = 256bit... 不对

# 更准确地说，cp.async 的向量化展开:
# 原: count=4 (4×float16 = 8字节)
# vector_size=4 → 合并后 count=4×4=16 (16×float16 = 32字节)
# 但 cp.async 最大传输 16 字节，所以实际 vector_size 受限

# VectorizePlanner 中的计算:
int vectorize_length = GetMaxCPAsyncVectorizeLength(per_call_bits);
// 对 4×float16 = 64bit per call
// 目标: 4, 8, 16 字节
// 64bit=8字节 → 可以合并到 16 字节 → vectorize_length = 2
// 所以实际最多 vec=2
```

### 向量化后的 PTX

```python
# ===== 优化后 (vec=2 时) =====
for vec in T.vectorized(2):
    idx = tid * 2 + vec
    T.ptx_cp_async(
        T.access_ptr(A_shared[idx], "w", 8),   # 合并: 8 个 float16
        T.access_ptr(A[idx], "r", 8),
        8,                                      # count=8 字节
        idx < 64                                # 边界检查 predicate
    )
```

### cp.async + 越界 predicate

当 safe_value = 0 时，`LegalizeSafeMemoryAccess` 会自动将边界检查转为 PTX predicate：

```python
# ===== 优化前（有越界风险） =====
@T.prim_func
def main(A: T.Tensor((16,), float16)):
    A_shared = T.alloc_buffer((16,), float16, scope="shared")
    for i in T.serial(4):
        T.ptx_cp_async(
            T.access_ptr(A_shared[i*4], "w", 4),
            T.access_ptr(A[i*4 + 10], "r", 4),   # i=2 时越界
            4
        )

# ===== 编译后 =====
# safe_value=0 → 用 PTX 条件拷贝 (第4个参数)
for i in T.serial(4):
    T.ptx_cp_async(
        T.access_ptr(A_shared[i*4], "w", 4),
        T.access_ptr(A[i*4 + 10], "r", 4),
        4,
        i*4 + 10 < 16   # ← predicate: 越界时零填充
    )
```

### 传输宽度有效性检查

```cpp
// 只有以下 CPAsync 传输宽度是有效的:
bool IsValidCPAsyncTransferBytes(int total_bytes) {
    return total_bytes == 4 || total_bytes == 8 || total_bytes == 16;
    // 4字节, 8字节, 16字节
}
// 如果合并后不在这些值中 → need_scalarize_ = true (标量化回退)
```

---

## 10. 场景六：原子操作向量化合并

### 场景描述

CUDA 支持向量化原子操作（`atomic_addx2`、`atomic_addx4`），`VectorizePlanner` 和 `TLVectorizer` 可以自动将标量原子操作合并为向量化版本。

### 支持的数据类型和向量宽度

```cpp
// src/transform/loop_vectorize.cc
int GetMaxAtomicVectorSize(DataType dtype, Target target) {
    if (dtype.is_float16() || dtype.is_bfloat16()) {
        return 2;   // atomic_addx2: 一次处理 2 个 fp16
    }
    if (dtype.is_float() && dtype.bits() == 32 &&
        TargetHasSMVersionGE(target, 90)) {
        return 4;   // atomic_addx4: 一次处理 4 个 fp32 (SM90+)
    }
    return 1;       // 不支持向量化
}
```

### 代码示例

```python
# ===== 优化前：标量原子操作 =====
@T.prim_func
def main(A: T.Tensor((128,), float16)):
    with T.Kernel(1, 1, threads=16) as (bx, by):
        tid = T.get_thread_binding()
        # 每个线程处理 8 个元素
        for vec in T.vectorized(8):
            T.atomic_add(A[tid * 8 + vec], 1.0)  # 8 次标量原子操作

# ===== VectorizePlanner 分析 =====
# float16 → max_atomic_vector_size = 2
# 所以原子操作约束 vector_size 最大为 2
# 最终 vector_size = GCD(8, 2) = 2

# ===== 编译后 =====
# 自动展开为外循环 + 内层向量化原子操作
for outer in T.serial(4):          # 外循环 4 次
    for vec in T.vectorized(2):    # 内层向量化原子操作 vec=2
        idx = tid * 8 + outer * 2 + vec
        T.atomic_addx2(A[idx], 1.0)  # ← 一次操作 2 个 float16
```

### SM90+ float32 原子操作（vec=4）

```python
# ===== 优化前 =====
@T.prim_func
def main(A: T.Tensor((128,), float32)):
    with T.Kernel(1, 1, threads=8) as (bx, by):
        tid = T.get_thread_binding()
        for vec in T.vectorized(4):
            T.atomic_add(A[tid * 4 + vec], 1.0)

# ===== 编译后 (SM90+, Hopper) =====
# float32 + SM90+ → max_atomic_vector_size = 4
# vec=4 → 直接使用 atomic_addx4
for vec in T.vectorized(4):
    T.atomic_addx4(A[tid * 4 + vec], 1.0)  # ← 一次原子操作 4×float32
```

### 与越界检测的交互

```python
# ===== 原始代码 =====
@T.prim_func
def main(A: T.Tensor((70,), float32)):
    with T.Kernel(1, 1, threads=16) as (bx, by):
        tid = T.get_thread_binding()
        for vec in T.vectorized(4):
            idx = tid * 4 + vec
            T.atomic_add(A[idx], 1.0)  # idx 可能 >= 70

# ===== 编译后 =====
# 原子操作越界时用 IfThenElse 包裹
for vec in T.vectorized(4):
    idx = tid * 4 + vec
    if idx < 70:                             # ← 边界检查
        T.atomic_addx4(A[idx], 1.0)          # ← 向量化原子操作
```

---

## 11. 场景七：标量化回退（无法向量化的情形）

### 场景描述

某些表达式无法被向量化，TLVectorizer 会自动回退到标量循环。

### 触发回退的情况

```cpp
// 以下情况会设置 need_scalarize_ = true
```

#### 情况 A：if_then_else 向量化后

```python
# ===== 无法向量化的代码 =====
@T.prim_func
def main(A: T.Tensor((32,), float32), cond: T.Tensor((32,), bool)):
    with T.Kernel(1, 1, threads=4) as (bx, by):
        tid = T.get_thread_binding()
        for vec in T.vectorized(8):
            # if_then_else 的条件是向量，不同 lane 可能走不同分支
            A[tid*8 + vec] = T.if_then_else(
                cond[tid*8 + vec],     # ← 向量条件
                A[tid*8 + vec] * 2.0,
                A[tid*8 + vec] * 0.5
            )

# ===== TLVectorizer 分析 =====
# cond 向量化后是向量类型
# if_then_else 的条件为向量 → 无法简单映射为 SIMD 操作
# → need_scalarize_ = true

# ===== 编译后：回退到标量循环 =====
for i in T.serial(8):   # ← 改为串行标量循环
    idx = tid * 8 + i
    A[idx] = T.if_then_else(
        cond[idx],
        A[idx] * 2.0,
        A[idx] * 0.5
    )
```

#### 情况 B：Broadcast 嵌套向量化

```python
# ===== 无法向量化的代码 =====
@T.prim_func
def main(A: T.Tensor((32,), float32)):
    with T.Kernel(1, 1, threads=4) as (bx, by):
        tid = T.get_thread_binding()
        for vec in T.vectorized(8):
            # Broadcast 中 value 已经是向量 → 需要标量化
            A[tid*8 + vec] = T.broadcast(A[tid*8 + vec], 2)  

# TLVectorizer:
#   VisitExpr_(const BroadcastNode *op):
#     if (value.dtype().is_scalable_or_fixed_length_vector())
#       → need_scalarize_ = true
```

#### 情况 C：不支持的 CallNode

```python
# ===== 无法向量化的代码 =====
@T.prim_func
def main(A: T.Tensor((32,), float32)):
    with T.Kernel(1, 1, threads=4) as (bx, by):
        tid = T.get_thread_binding()
        for vec in T.vectorized(8):
            # call_extern 不是向量化的
            A[tid*8 + vec] = T.call_extern("my_custom_op", A[tid*8 + vec])

# ===== 编译后 =====
# CallNode 检查 op_vectorizable_ 属性
# 如果 op 没有标记 TVectorizable → 无法向量化
# → 所有参数即使向量化也会触发标量化回退
for i in T.serial(8):
    A[tid*8 + i] = T.call_extern("my_custom_op", A[tid*8 + i])
```

#### 情况 D：cp.async 合并后宽度无效

```python
# VectorizePlanner 分析:
# cp.async 传输宽度必须是 4/8/16 字节
# 如果合并后 total_bytes 不在这些值中
# → need_scalarize_ = true
# → 回退到多个标量 cp.async
```

### 回退机制源码

```cpp
// src/transform/vectorize_loop.cc
class TLVectorizer {
    bool need_scalarize_{false};
    
    // 标量化回退
    Stmt Scalarize(Stmt stmt) {
        Var idx(var_->name_hint + "_s", var_->dtype);
        stmt = Substitute(stmt, {{var_, idx}});
        return For(idx, 0, var_lanes_, ForKind::kSerial, stmt);
    }
};
```

---

## 12. 场景八：嵌套循环向量化（外循环 + 内层向量化）

### 场景描述

当向量化范围大于 `vector_size` 时，自动拆分为外循环 + 内层向量化循环。

### 代码示例

```python
# ===== 优化前 =====
@T.prim_func
def main(A: T.Tensor((128,), float32)):
    with T.Kernel(1, 1, threads=4) as (bx, by):
        tid = T.get_thread_binding()
        for vec in T.vectorized(32):       # ← 范围 32，但一次只能处理 4 个
            A[tid*32 + vec] = 1.0

# ===== VectorizePlanner 分析 =====
# float32 → max_vector_size = 128/32 = 4
# extent = 32, 取 GCD(32, 4) = 4
# → vector_size = 4

# ===== 编译后：自动拆分 =====
# VectorizeRewriter 自动拆分为外循环 + 内层向量化
for outer in T.serial(8):               # 外循环 32/4 = 8 次
    for vec in T.vectorized(4):          # 内层向量化 4 个元素
        A[tid*32 + outer*4 + vec] = 1.0
```

### 自动拆分逻辑

```cpp
// src/transform/loop_vectorize.cc
// VectorizeRewriter::VisitStmt_
if (extent == vector_size_) {
    // 正好等于向量宽度 → 直接标记为 kVectorized
    fnode.CopyOnWrite()->kind = ForKind::kVectorized;
    return fnode;
} else {
    // 需要拆分
    // inner: kVectorized, extent = vector_size_
    // outer: kSerial, extent = extent / vector_size_
    Var inner_var("vec");
    Var outer_var(old_var->name_hint);
    // 变量映射: old_var = outer_var * vector_size_ + inner_var
    vmap.Set(fnode->loop_var, outer_var * vector_size_ + inner_var);
    Stmt body = Substitute(fnode->body, vmap);
    body = For(inner_var, 0, vector_size_, ForKind::kVectorized, body);
    body = For(outer_var, 0, extent / vector_size_, ForKind::kSerial, body, ...);
    return body;
}
```

### 多种循环类型的处理

```python
# 如果外层循环最初是 T.Parallel:
@T.prim_func
def main(A: T.Tensor((128,), float32)):
    with T.Kernel(1, 1, threads=4) as (bx, by):
        tid = T.get_thread_binding()
        for i in T.Parallel(8):            # 外层 Parallel
            for vec in T.vectorized(4):     # 内层向量化
                A[tid*32 + i*4 + vec] = 1.0

# 编译后外层 Parallel 自动降级为 Serial
# （向量化后 Parallel 语义不再有意义）
```

---

## 13. 场景九：T.vectorized 与 T.Parallel 配合

### 场景描述

在 TileLang 中，常见的模式是外层用 `T.Parallel` 做 SIMT 线程映射，内层用 `T.vectorized` 做向量化。

### 推荐模式

```python
# ===== 推荐写法 =====
@T.prim_func
def main(A: T.Tensor((1024,), float32), B: T.Tensor((1024,), float32),
         C: T.Tensor((1024,), float32)):
    with T.Kernel(1, 1, threads=256) as (bx, by):
        tid = T.get_thread_binding()
        # 外层 T.Parallel: SIMT 线程并行
        for i in T.Parallel(4):
            # 内层 T.vectorized: 向量化
            for vec in T.vectorized(4):
                idx = tid * 16 + i * 4 + vec
                C[idx] = A[idx] + B[idx]

# 解释:
# - 256 个线程
# - 每个线程处理 4 个 "并行块" (i=0..3)
# - 每个并行块内 4 个向量化元素 (vec=0..3)
# - 总共: 256 × 4 × 4 = 4096 元素
# - 但实际上这里的 1024 / (256*4) = 1，所以 i 只有 1
# 更合理:
#   threads=256, 每个线程处理 4 个元素
#   256 × 4 = 1024
#   T.Parallel(1) → 不需要外层并行
```

### 正确的线程映射模式

```python
# ===== 更实用的模式 =====
@T.prim_func
def main(A: T.Tensor((1024,), float32)):
    with T.Kernel(1, 1, threads=256) as (bx, by):
        tid = T.get_thread_binding()
        # 每个线程直接向量化处理 4 个元素
        # 256 线程 × 4 元素 = 1024
        for vec in T.vectorized(4):
            A[tid * 4 + vec] = 1.0

# 生成的 PTX 伪代码:
# tid = threadIdx.x
# STG.128 [A + tid*16], {1.0, 1.0, 1.0, 1.0}
# ← 256 个线程，每个存储 16 字节 = 4096 字节 = 1024 × float32 ✓
```

### 大 tile 的拆分

```python
# ===== 每个线程处理更多元素 =====
@T.prim_func
def main(A: T.Tensor((2048,), float32)):
    with T.Kernel(1, 1, threads=256) as (bx, by):
        tid = T.get_thread_binding()
        # 每个线程处理 8 个元素
        # 256 × 8 = 2048
        for i in T.serial(2):              # 外循环 2 次
            for vec in T.vectorized(4):     # 内循环 4 个元素
                A[tid * 8 + i * 4 + vec] = 1.0
```

---

## 14. 场景十：T.vectorized 与 swizzle 布局的协同

### 场景描述

在共享内存的 GEMM 操作中，向量化宽度决定了 swizzle 的粒度，以避免 bank conflict。

### 相关代码

```python
# tilelang/tileop/gemm/gemm_wgmma.py
def infer_shared_layout(self, continuity: int):
    """根据连续性和向量化宽度推断 swizzle 布局"""
    vectorized_size = 128 // self.in_dtype.bits
    # continuity = 在连续维度上的元素数
    
    if continuity % (vectorized_size * 8) == 0:
        return make_full_bank_swizzled_layout    # 128B swizzle
    elif continuity % (vectorized_size * 4) == 0:
        return make_half_bank_swizzled_layout    # 64B swizzle
    elif continuity % (vectorized_size * 2) == 0:
        return make_quarter_bank_swizzled_layout # 32B swizzle
    else:
        return make_linear_layout                # 无 swizzle
```

### 示例

```python
# ===== float16 矩阵乘法的共享内存布局 =====
# vectorized_size = 128 / 16 = 8 (每个向量化操作 8 个 float16)

# 如果 continuity = 64 (64 个 float16 = 128 字节)
# → 128B Full Bank Swizzle
#   continuity % (8 * 8) == 64 % 64 == 0 ✓

# 如果 continuity = 32 (32 个 float16 = 64 字节)
# → 64B Half Bank Swizzle
#   continuity % (8 * 4) == 32 % 32 == 0 ✓

# 如果 continuity = 16 (16 个 float16 = 32 字节)
# → 32B Quarter Bank Swizzle
#   continuity % (8 * 2) == 16 % 16 == 0 ✓
```

### 意义

向量化宽度和 swizzle 的协同确保：
1. 每次向量化加载/存储都是对齐的
2. 共享内存 bank 访问均匀分布，避免冲突
3. Tensor Core 的数据供给达到最大带宽

---

## 15. 优化建议与最佳实践

### 15.1 何时使用 T.vectorized

| 场景 | 推荐使用 | 原因 |
|------|---------|------|
| **连续内存访问** | ✅ 使用 | 自然适合向量化 |
| **混合精度计算** | ✅ 使用 | DecoupleTypeCast 会自动优化 |
| **元素级操作** | ✅ 使用 | 加、乘、激活函数等 |
| **条件分支** | ❌ 避免 | if_then_else 会触发标量化回退 |
| **非连续索引** | ❌ 避免 | `A[vec*2]` 无法向量化 |
| **自定义外部函数** | ❌ 避免 | call_extern 不向量化 |

### 15.2 选择合适的向量宽度

```python
# 推荐: 让编译器自动决定
# 直接用 T.vectorized 并指定足够大的范围
for vec in T.vectorized(16):   # 编译器自动确定最佳 vector_size
    ...

# 不推荐: 强制固定宽度（除非有特殊原因）
# 编译器可能无法优化
```

### 15.3 避免导致标量化回退的模式

```python
# ❌ 不好的写法：会标量化回退
@T.prim_func
def bad(A: T.Tensor((128,), float32), cond: T.Tensor((128,), bool)):
    with T.Kernel(1, 1, threads=32) as (bx, by):
        tid = T.get_thread_binding()
        for vec in T.vectorized(4):
            A[tid*4 + vec] = T.if_then_else(
                cond[tid*4 + vec],      # ← 条件分支导致标量化
                A[tid*4 + vec] * 2.0,
                A[tid*4 + vec] * 0.5
            )

# ✅ 好的写法：用 T.where 或先计算后选择
@T.prim_func
def good(A: T.Tensor((128,), float32), mask: T.Tensor((128,), float32)):
    with T.Kernel(1, 1, threads=32) as (bx, by):
        tid = T.get_thread_binding()
        for vec in T.vectorized(4):
            # 用乘法掩码替代条件分支
            A[tid*4 + vec] = A[tid*4 + vec] * (2.0 * mask[tid*4 + vec] + 0.5 * (1 - mask[tid*4 + vec]))
            # 或者将条件提取到循环外
```

### 15.4 合理利用越界安全检测

```python
# ✅ 充分利用此特性可以省去手动边界检查
@T.prim_func
def main(A: T.Tensor((100,), float32)):  # 100 不能被 4 整除
    with T.Kernel(1, 1, threads=25) as (bx, by):
        tid = T.get_thread_binding()
        for vec in T.vectorized(4):
            idx = tid * 4 + vec
            # 不需要手动 if idx < 100
            A[idx] = 1.0
            # LegalizeSafeMemoryAccess 自动处理
            # 当 idx >= 100 时跳过写入
```

### 15.5 性能调优检查清单

- [ ] **循环范围**是否足够大（至少 4 个元素）？
- [ ] **数据类型**是否一致？混合精度需要 DecoupleTypeCast
- [ ] **内存访问**是否连续？（`A[base + vec]` 模式）
- [ ] **是否有条件分支**导致标量化回退？
- [ ] **是否有外部函数调用**阻止向量化？
- [ ] **Global 独占**时是否能启用 256-bit 向量化？
- [ ] **原子操作**是否能利用 hardware 向量化（atomic_addx2/x4）？

---

## 16. 附录：关键配置项

### 编译时配置

```python
# 通过 PassContext 控制向量化行为
with tilelang.transform.PassContext(config={
    # 完全禁用向量化
    "tir.disable_vectorize": True,
    
    # 禁用 256-bit 向量化（只用 128-bit）
    "tl.disable_vectorize_256": True,
    
    # 禁用安全内存检查
    "tl.disable_safe_memory_legalize": True,
    
    # 禁用 OOB 警告（但不禁用守卫插入）
    "tl.disable_out_of_bound_warning": True,
    
    # 启用 VectorizePlanner 详细日志
    "tl.vectorize_planner_verbose": True,
}):
    kernel = matmul_relu.compile(a, b)
```

### 向量化风格选择

```python
# 在 T.vectorized 中通过 annotations 传递
for vec in T.vectorized(4, annotations={
    "pragma_vectorize": True,      # 提示编译器向量化
    # 其他 TVM 向量化相关 annotation
}):
    ...
```

### 相关的环境变量

| 变量 | 作用 |
|------|------|
| `TVM_LOG_DEBUG` | 开启 TVM 调试日志，可看到向量化决策 |
| `TILELANG_DEBUG` | 开启 TileLang 调试模式 |

---

> **总结**：`T.vectorized` 是 TileLang 中实现高性能内存访问的关键机制。它通过 **VectorizePlanner 自动规划最佳向量宽度**，配合 **DecoupleTypeCast 解耦混合精度**、**LegalizeSafeMemoryAccess 自动越界保护**、**TLVectorizer 的智能标量化回退**等机制，在保证正确性的前提下最大化内存带宽利用率。理解这些机制有助于写出既高效又安全的 GPU kernel。
