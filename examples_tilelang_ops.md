# TileLang OP 列表

## 同步语义 OP

| OP 名称 | 是否支持 | OP 含义 |
|---|---|---|
| T.sync_threads | 否 | 线程同步 |
| T.barrier_arrive | 否 | 线程屏障到达 |
| T.barrier_wait | 否 | 线程屏障等待 |
| T.mbarrier_arrive | 否 | 多线程屏障到达 |
| T.mbarrier_wait_parity | 否 | 多线程屏障等待奇偶 |
| T.sync_grid | 否 | 栅格同步 |
| T.tcgen05_after_thread_sync | 否 | tcgen05 线程同步后标记 |
| T.tcgen05_before_thread_sync | 否 | tcgen05 线程同步前标记 |
| T.tcgen05_mma_arrive | 否 | tcgen05 MMA 到达 |
| T.wait_wgmma | 否 | 等待 wgmma 完成 |
| T.cp_async_barrier_noinc | 否 | cp.async 屏障（不递增） |

## 内存申请语义 OP

| OP 名称 | 是否支持 | OP 含义 |
|---|---|---|
| T.alloc_fragment | 是 | 分配片段寄存器 |
| T.alloc_shared | 是 | 分配共享内存 |
| T.alloc_barrier | 否 | 分配屏障对象 |
| T.alloc_buffer | 否 | 分配缓冲区 |
| T.alloc_cluster_barrier | 否 | 分配集群屏障 |
| T.alloc_global | 否 | 分配全局内存 |
| T.alloc_local | 否 | 分配本地内存 |
| T.alloc_reducer | 否 | 分配归约中间值 |
| T.alloc_tmem | 否 | 分配tmem临时缓冲 |
| T.alloc_var | 否 | 分配变量空间 |
| T.make_tensor | 否 | 创建张量 |
| T.empty | 否 | 空张量 |

## 数据搬运语义 OP

| OP 名称 | 是否支持 | OP 含义 |
|---|---|---|
| T.copy | 是 | 拷贝数据 |
| T.tma_copy | 否 | TMA 复制 |

## 计算语义 OP

| OP 名称 | 是否支持 | OP 含义 |
|---|---|---|
| T.gemm | 是 | 矩阵乘累加 |
| T.reduce_absmax | 是 | 绝对值最大归约 |
| T.reduce_max | 是 | 最大值归约 |
| T.reduce_sum | 是 | 求和归约 |
| T.sigmoid | 是 | sigmoid 函数 |
| T.cumsum | 是 | 累加和 |
| T.atomic_add | 是 | 原子加操作 |
| T.atomic_addx4 | 是 | 四元原子加 |
| T.if_then_else | 是 | 条件表达式 |
| T.reshape | 是 | 重塑张量 |
| T.symbolic | 是 | 符号变量 |
| T.dynamic | 是 | 动态维度或变量 |
| T.ceildiv | 是 | 向上取整除 |
| T.fill | 是 | 填充常量 |
| T.clear | 是 | 清零或重置数据 |
| T.gemm_sp | 否 | 稀疏矩阵乘累加 |
| T.gemm_sp_v2 | 否 | 稀疏矩阵乘累加 v2 |
| T.wgmma_gemm | 否 | WGMMA GEMM |
| T.tcgen05_gemm | 否 | tcgen05 GEMM 操作 |
| T.max | 否 | 最大值 |
| T.min | 否 | 最小值 |
| T.exp | 否 | 指数 |
| T.exp2 | 否 | 2的指数 |
| T.log | 否 | 自然对数 |
| T.log2 | 否 | log2 |
| T.sin | 否 | 正弦函数 |
| T.cos | 否 | 余弦函数 |
| T.rsqrt | 否 | 倒数平方根 |
| T.dp4a | 否 | dp4a 指令 |
| T.cast | 否 | 类型转换 |
| T.clamp | 否 | 截断到范围 |
| T.reinterpret | 否 | 重新解释类型 |
| T.replace | 否 | 替换表达式 |
| T.view | 否 | 视图数据 |
| T.evaluate | 否 | 表达式求值 |
| T.const | 否 | 常量 |
| T.index_to_coordinates | 否 | 从索引到坐标转换 |
| T.rng_init | 否 | 随机数初始化 |
| T.rng_rand | 否 | 随机数生成 |
| T.shift_left | 否 | 左移位 |
| T.floordiv | 否 | 向下除 |
| T.floormod | 否 | 向下取模 |
| T.comm_reducer | 否 | 通信归约操作 |
| T.finalize_reducer | 否 | 完成归约 |
| T.tvm_thread_allreduce | 否 | TVM 线程归约 |
| T.tvm_warp_shuffle | 否 | TVM warp shuffle |
| T.c2d_im2col | 否 | 卷积 im2col 转换 |
| T.call_extern | 否 | 调用外部函数 |
| T.access_ptr | 否 | 访问指针转换 |

## 定义类（数据类型 OP）

| OP 名称 | 是否支持 | OP 含义 |
|---|---|---|
| T.Tensor | 是 | 张量类型 |
| T.Buffer | 是 | Buffer类型声明或缓存对象 |
| T.dtype | 是 | 数据类型 |
| T.infinity | 是 | 无穷大常量 |
| T.prim_func | 是 | 原语函数定义 |
| T.Kernel | 是 | 内核定义上下文 |
| T.Parallel | 是 | 并行循环构造 |
| T.Pipelined | 是 | 流水线循环构造 |
| T.serial | 是 | 串行循环 |
| T.macro | 是 | 宏定义 |
| T.FragmentBuffer | 否 | 片段级缓冲区 |
| T.SharedBuffer | 否 | 共享缓冲区 |
| T.StridedTensor | 否 | 带stride的张量类型 |
| T.Ref | 否 | 引用类型 |
| T.ptr | 否 | 指针类型 |
| T.Layout | 否 | 布局说明 |
| T.bfloat16 | 否 | bfloat16 数据类型 |
| T.float | 否 | float32 类型 |
| T.float16 | 否 | float16 类型 |
| T.float32 | 否 | float32 类型 |
| T.float8_e4m3fn | 否 | float8 e4m3fn 类型 |
| T.float8_e4m3fnuz | 否 | float8 e4m3fnuz 类型 |
| T.float8_e5m2 | 否 | float8 e5m2 类型 |
| T.float8_e5m2fnuz | 否 | float8 e5m2fnuz 类型 |
| T.half | 否 | float16 别名 |
| T.int8 | 否 | int8 类型 |
| T.int16 | 否 | int16 类型 |
| T.int32 | 否 | int32 类型 |
| T.uint8 | 否 | uint8 类型 |
| T.uint16 | 否 | uint16 类型 |
| T.uint32 | 否 | uint32 类型 |
| T.uint64 | 否 | uint64 类型 |
| T.bool | 否 | 布尔类型 |
| T.contiguous | 否 | 连续内存布局 |
| T.unroll | 否 | 展开循环 |
| T.vectorized | 否 | 向量化 |
| T.Persistent | 否 | 持久存储或生命周期指示 |
| T.GemmWarpPolicy | 否 | GEMM warp级策略枚举 |
| T.thread_binding | 否 | 线程绑定 |
| T.block | 否 | 块范围定义 |
| T.block_attr | 否 | 块属性注释 |
| T.block_rank_in_cluster | 否 | 集群中块排名 |
| T.get_block_binding | 否 | 获取块绑定坐标 |
| T.get_thread_binding | 否 | 获取线程绑定坐标 |
| T.import_source | 否 | 导入外部源 |
| T.annotate_consumer_reg_alloc | 否 | 标注消费者寄存器分配规则 |
| T.annotate_layout | 否 | 标注布局属性 |
| T.annotate_producer_reg_dealloc | 否 | 标注生产者寄存器释放 |
| T.attr | 否 | 属性注释 |
| T.assume | 否 | 设定编译器假设 |
| T.disable_warp_group_reg_alloc | 否 | 禁用warp组寄存器分配 |
| T.no_set_max_nreg | 否 | No set max register 限制 |
| T.set_max_nreg | 否 | 设置最大寄存器数 |
| T.use_swizzle | 否 | 使用地址扰动 |
| T.ws | 否 | 工作集设置 |
| T.writes | 否 | 写入依赖 |
