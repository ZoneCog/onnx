<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX Runtime Architecture

This document provides detailed architecture diagrams specifically focused on ONNX runtime execution patterns and backend integration.

## Runtime Execution Flow

```mermaid
sequenceDiagram
    participant App as Application
    participant Session as InferenceSession
    participant Provider as ExecutionProvider
    participant Allocator as MemoryAllocator
    participant Kernel as OpKernel

    App->>Session: CreateSession(model_path)
    Session->>Session: Load and validate model
    Session->>Provider: RegisterProvider()
    Provider->>Provider: Initialize device resources
    
    App->>Session: Run(inputs)
    Session->>Allocator: Allocate input tensors
    Allocator-->>Session: Return tensor buffers
    
    loop For each node in execution order
        Session->>Provider: GetKernel(op_type)
        Provider-->>Session: Return kernel instance
        Session->>Kernel: Compute(inputs, outputs)
        Kernel->>Allocator: Allocate output tensors
        Allocator-->>Kernel: Return output buffers
        Kernel->>Kernel: Execute operation
        Kernel-->>Session: Return computed outputs
    end
    
    Session-->>App: Return final outputs
    
    App->>Session: Release resources
    Session->>Allocator: Deallocate tensors
    Session->>Provider: Cleanup device resources
```

## Memory Management Architecture

```mermaid
graph TB
    subgraph MemorySystem["ONNX Runtime Memory System"]
        subgraph Allocators["Memory Allocators"]
            CPUAllocator["CPU Allocator<br/>System memory"]
            CUDAAllocator["CUDA Allocator<br/>GPU memory"]
            PinnedAllocator["Pinned Memory Allocator<br/>CUDA pinned memory"]
            ArenaAllocator["Arena Allocator<br/>Pre-allocated pools"]
        end
        
        subgraph MemoryPools["Memory Pools"]
            CPUPool["CPU Memory Pool<br/>Reusable buffers"]
            GPUPool["GPU Memory Pool<br/>Device memory cache"]
            SharedPool["Shared Memory Pool<br/>Cross-device sharing"]
        end
        
        subgraph MemoryPlan["Memory Planning"]
            LifetimeAnalysis["Tensor Lifetime<br/>Analysis"]
            MemoryReuse["Memory Reuse<br/>Optimization"]
            MemoryPattern["Memory Access<br/>Patterns"]
            AllocationStrategy["Allocation<br/>Strategy"]
        end
        
        subgraph GarbageCollection["Garbage Collection"]
            RefCounting["Reference<br/>Counting"]
            EagerCleanup["Eager<br/>Cleanup"]
            DelayedCleanup["Delayed<br/>Cleanup"]
            MemoryPressure["Memory Pressure<br/>Monitoring"]
        end
    end
    
    Allocators --> MemoryPools
    MemoryPools --> MemoryPlan
    MemoryPlan --> GarbageCollection
    
    %% Detailed connections
    LifetimeAnalysis --> MemoryReuse
    MemoryReuse --> AllocationStrategy
    AllocationStrategy --> CPUPool
    AllocationStrategy --> GPUPool
    
    RefCounting --> EagerCleanup
    MemoryPressure --> DelayedCleanup
    
    %% Styling
    classDef allocator fill:#e8f5e8
    classDef pool fill:#e3f2fd
    classDef plan fill:#fff3e0
    classDef gc fill:#f3e5f5
    
    class CPUAllocator,CUDAAllocator,PinnedAllocator,ArenaAllocator,Allocators allocator
    class CPUPool,GPUPool,SharedPool,MemoryPools pool
    class LifetimeAnalysis,MemoryReuse,MemoryPattern,AllocationStrategy,MemoryPlan plan
    class RefCounting,EagerCleanup,DelayedCleanup,MemoryPressure,GarbageCollection gc
```

## Parallel Execution Architecture

```mermaid
graph TB
    subgraph ParallelExecution["Parallel Execution Architecture"]
        subgraph ThreadManagement["Thread Management"]
            MainThread["Main Thread<br/>Orchestration"]
            WorkerThreads["Worker Threads<br/>Parallel execution"]
            IOThreads["I/O Threads<br/>Async data loading"]
            DeviceThreads["Device Threads<br/>GPU operations"]
        end
        
        subgraph TaskScheduling["Task Scheduling"]
            TaskQueue["Task Queue<br/>Pending operations"]
            Scheduler["Task Scheduler<br/>Work distribution"]
            LoadBalancer["Load Balancer<br/>Resource optimization"]
            DependencyGraph["Dependency Graph<br/>Execution order"]
        end
        
        subgraph ExecutionModes["Execution Modes"]
            Sequential["Sequential Mode<br/>Single-threaded"]
            Parallel["Parallel Mode<br/>Multi-threaded"]
            Pipeline["Pipeline Mode<br/>Overlapped execution"]
            Async["Async Mode<br/>Non-blocking calls"]
        end
        
        subgraph Synchronization["Synchronization"]
            Barriers["Thread Barriers<br/>Sync points"]
            Mutexes["Mutexes<br/>Critical sections"]
            Atomics["Atomic Operations<br/>Lock-free updates"]
            Events["CUDA Events<br/>GPU synchronization"]
        end
    end
    
    ThreadManagement --> TaskScheduling
    TaskScheduling --> ExecutionModes
    ExecutionModes --> Synchronization
    
    %% Cross-connections
    MainThread --> Scheduler
    WorkerThreads --> TaskQueue
    DeviceThreads --> Events
    DependencyGraph --> LoadBalancer
    
    %% Styling
    classDef thread fill:#e8f5e8
    classDef schedule fill:#e3f2fd
    classDef mode fill:#fff3e0
    classDef sync fill:#f3e5f5
    
    class MainThread,WorkerThreads,IOThreads,DeviceThreads,ThreadManagement thread
    class TaskQueue,Scheduler,LoadBalancer,DependencyGraph,TaskScheduling schedule
    class Sequential,Parallel,Pipeline,Async,ExecutionModes mode
    class Barriers,Mutexes,Atomics,Events,Synchronization sync
```

## Provider Selection and Fallback

```mermaid
flowchart TD
    ModelLoad["Model Loading"]
    
    subgraph ProviderSelection["Provider Selection Process"]
        ProviderDiscovery["Provider Discovery<br/>Available providers"]
        CapabilityCheck["Capability Check<br/>Can handle operations?"]
        PerformanceRanking["Performance Ranking<br/>Provider priority"]
        ResourceCheck["Resource Check<br/>Hardware availability"]
    end
    
    subgraph DecisionEngine["Decision Engine"]
        OpPlacement["Operator Placement<br/>Per-op provider selection"]
        GraphPartitioning["Graph Partitioning<br/>Provider boundaries"]
        FallbackStrategy["Fallback Strategy<br/>Error handling"]
        CostModel["Cost Model<br/>Performance estimation"]
    end
    
    subgraph ProviderTypes["Available Providers"]
        subgraph HighPerf["High Performance"]
            TensorRT["TensorRT<br/>NVIDIA optimization"]
            DNNL["OneDNN<br/>Intel optimization"]
            DirectML["DirectML<br/>Windows GPU"]
        end
        
        subgraph Standard["Standard"]
            CUDA["CUDA<br/>NVIDIA GPU"]
            CPU["CPU<br/>Default fallback"]
            OpenMP["OpenMP<br/>CPU parallelization"]
        end
        
        subgraph Specialized["Specialized"]
            Custom["Custom<br/>User-defined"]
            Experimental["Experimental<br/>Research providers"]
        end
    end
    
    subgraph FallbackChain["Fallback Chain"]
        Primary["Primary Provider<br/>Best performance"]
        Secondary["Secondary Provider<br/>Fallback option"]
        Default["Default Provider<br/>CPU guarantee"]
        Error["Error Handler<br/>Unsupported ops"]
    end
    
    ModelLoad --> ProviderSelection
    ProviderSelection --> DecisionEngine
    DecisionEngine --> ProviderTypes
    ProviderTypes --> FallbackChain
    
    %% Decision flow
    OpPlacement --> Primary
    Primary --> Secondary
    Secondary --> Default
    Default --> Error
    
    %% Capability connections
    CapabilityCheck --> HighPerf
    CapabilityCheck --> Standard
    CapabilityCheck --> Specialized
    
    %% Styling
    classDef load fill:#e8f5e8
    classDef selection fill:#e3f2fd
    classDef decision fill:#fff3e0
    classDef highperf fill:#c8e6c9
    classDef standard fill:#bbdefb
    classDef specialized fill:#f8bbd9
    classDef fallback fill:#ffecb3
    
    class ModelLoad load
    class ProviderDiscovery,CapabilityCheck,PerformanceRanking,ResourceCheck,ProviderSelection selection
    class OpPlacement,GraphPartitioning,FallbackStrategy,CostModel,DecisionEngine decision
    class TensorRT,DNNL,DirectML,HighPerf highperf
    class CUDA,CPU,OpenMP,Standard standard
    class Custom,Experimental,Specialized specialized
    class Primary,Secondary,Default,Error,FallbackChain fallback
```

## Cross-Platform Runtime Architecture

```mermaid
graph TB
    subgraph CrossPlatform["Cross-Platform Runtime Architecture"]
        subgraph PlatformAbstraction["Platform Abstraction Layer"]
            OSAbstraction["OS Abstraction<br/>Windows/Linux/macOS"]
            HardwareAbstraction["Hardware Abstraction<br/>CPU/GPU/NPU"]
            DriverAbstraction["Driver Abstraction<br/>CUDA/OpenCL/DirectX"]
        end
        
        subgraph RuntimeCore["Runtime Core"]
            ExecutionEngine["Execution Engine<br/>Platform-agnostic"]
            MemoryManager["Memory Manager<br/>Unified interface"]
            ThreadManager["Thread Manager<br/>Cross-platform threading"]
        end
        
        subgraph PlatformSpecific["Platform-Specific Implementations"]
            subgraph Windows["Windows"]
                D3D12["Direct3D 12"]
                WinRT["Windows Runtime"]
                UWP["UWP Support"]
            end
            
            subgraph Linux["Linux"]
                CUDA_Linux["CUDA"]
                OpenCL_Linux["OpenCL"]
                ROCm["AMD ROCm"]
            end
            
            subgraph Mobile["Mobile Platforms"]
                Android["Android<br/>NNAPI"]
                iOS["iOS<br/>Core ML"]
                ARM["ARM Mali GPU"]
            end
        end
        
        subgraph DeviceSupport["Device Support"]
            CPUSupport["CPU Support<br/>x86/ARM/RISC-V"]
            GPUSupport["GPU Support<br/>NVIDIA/AMD/Intel"]
            NPUSupport["NPU Support<br/>Dedicated AI chips"]
            EdgeSupport["Edge Support<br/>IoT devices"]
        end
    end
    
    PlatformAbstraction --> RuntimeCore
    RuntimeCore --> PlatformSpecific
    PlatformSpecific --> DeviceSupport
    
    %% Cross-connections
    OSAbstraction --> Windows
    OSAbstraction --> Linux
    OSAbstraction --> Mobile
    
    HardwareAbstraction --> CPUSupport
    HardwareAbstraction --> GPUSupport
    HardwareAbstraction --> NPUSupport
    
    %% Styling
    classDef abstraction fill:#e8f5e8
    classDef core fill:#e3f2fd
    classDef windows fill:#bbdefb
    classDef linux fill:#c8e6c9
    classDef mobile fill:#f8bbd9
    classDef device fill:#ffecb3
    
    class OSAbstraction,HardwareAbstraction,DriverAbstraction,PlatformAbstraction abstraction
    class ExecutionEngine,MemoryManager,ThreadManager,RuntimeCore core
    class D3D12,WinRT,UWP,Windows windows
    class CUDA_Linux,OpenCL_Linux,ROCm,Linux linux
    class Android,iOS,ARM,Mobile mobile
    class CPUSupport,GPUSupport,NPUSupport,EdgeSupport,DeviceSupport device
```

## Runtime Performance Optimization

```mermaid
graph TB
    subgraph PerformanceOptimization["Runtime Performance Optimization"]
        subgraph ProfilingSystem["Profiling System"]
            CPUProfiler["CPU Profiler<br/>Execution timing"]
            MemoryProfiler["Memory Profiler<br/>Usage tracking"]
            GPUProfiler["GPU Profiler<br/>Kernel timing"]
            NetworkProfiler["Network Profiler<br/>Model structure"]
        end
        
        subgraph OptimizationStrategies["Optimization Strategies"]
            StaticOptimization["Static Optimization<br/>Compile-time"]
            DynamicOptimization["Dynamic Optimization<br/>Runtime adaptation"]
            CacheOptimization["Cache Optimization<br/>Data locality"]
            KernelFusion["Kernel Fusion<br/>Reduce overhead"]
        end
        
        subgraph AdaptiveOptimization["Adaptive Optimization"]
            RuntimeAdaptation["Runtime Adaptation<br/>Dynamic tuning"]
            LoadBalancing["Load Balancing<br/>Resource distribution"]
            PowerManagement["Power Management<br/>Energy efficiency"]
            ThermalThrottling["Thermal Throttling<br/>Temperature control"]
        end
        
        subgraph Benchmarking["Benchmarking & Monitoring"]
            PerformanceMetrics["Performance Metrics<br/>Latency/throughput"]
            ResourceUtilization["Resource Utilization<br/>CPU/GPU/Memory"]
            PowerConsumption["Power Consumption<br/>Energy monitoring"]
            SystemHealth["System Health<br/>Overall status"]
        end
        
        subgraph OptimizationTechniques["Optimization Techniques"]
            Quantization["Quantization<br/>Reduced precision"]
            Pruning["Model Pruning<br/>Remove redundancy"]
            OperatorFusion["Operator Fusion<br/>Combine operations"]
            MemoryOptimization["Memory Optimization<br/>Reduce footprint"]
        end
    end
    
    ProfilingSystem --> OptimizationStrategies
    OptimizationStrategies --> AdaptiveOptimization
    AdaptiveOptimization --> Benchmarking
    Benchmarking --> OptimizationTechniques
    
    %% Cross-connections
    CPUProfiler --> StaticOptimization
    MemoryProfiler --> CacheOptimization
    GPUProfiler --> KernelFusion
    RuntimeAdaptation --> LoadBalancing
    
    %% Styling
    classDef profiling fill:#e8f5e8
    classDef optimization fill:#e3f2fd
    classDef adaptive fill:#fff3e0
    classDef benchmark fill:#f3e5f5
    classDef techniques fill:#fce4ec
    
    class CPUProfiler,MemoryProfiler,GPUProfiler,NetworkProfiler,ProfilingSystem profiling
    class StaticOptimization,DynamicOptimization,CacheOptimization,KernelFusion,OptimizationStrategies optimization
    class RuntimeAdaptation,LoadBalancing,PowerManagement,ThermalThrottling,AdaptiveOptimization adaptive
    class PerformanceMetrics,ResourceUtilization,PowerConsumption,SystemHealth,Benchmarking benchmark
    class Quantization,Pruning,OperatorFusion,MemoryOptimization,OptimizationTechniques techniques
```

## Runtime Error Handling and Recovery

```mermaid
stateDiagram-v2
    [*] --> Initializing
    
    Initializing --> Ready : Success
    Initializing --> InitError : Failure
    
    Ready --> Executing : Run()
    Ready --> Cleanup : Release()
    
    Executing --> Ready : Success
    Executing --> RuntimeError : Execution failure
    Executing --> MemoryError : Out of memory
    Executing --> DeviceError : Hardware failure
    
    RuntimeError --> Recovering : Retry strategy
    MemoryError --> MemoryRecovery : Free resources
    DeviceError --> DeviceRecovery : Reset device
    
    Recovering --> Ready : Recovery success
    Recovering --> FatalError : Recovery failed
    
    MemoryRecovery --> Ready : Memory freed
    MemoryRecovery --> FatalError : Cannot recover
    
    DeviceRecovery --> Ready : Device reset
    DeviceRecovery --> FatalError : Device unavailable
    
    InitError --> [*] : Cannot initialize
    FatalError --> [*] : Unrecoverable
    Cleanup --> [*] : Resources released
    
    note right of Recovering
        - Retry with different provider
        - Fallback to CPU
        - Reduce batch size
    end note
    
    note right of MemoryRecovery
        - Free unused tensors
        - Garbage collection
        - Reduce memory usage
    end note
    
    note right of DeviceRecovery
        - Reset GPU context
        - Reinitialize drivers
        - Switch to backup device
    end note
```

## Summary

This runtime architecture documentation provides comprehensive coverage of:

1. **Runtime Execution Flow**: Detailed sequence of operations during model inference
2. **Memory Management**: Sophisticated memory allocation and optimization strategies
3. **Parallel Execution**: Multi-threaded and multi-device execution patterns
4. **Provider Selection**: Intelligent backend selection with fallback mechanisms
5. **Cross-Platform Support**: Architecture for supporting diverse platforms and devices
6. **Performance Optimization**: Runtime optimization techniques and adaptive strategies
7. **Error Handling**: Robust error recovery and fallback mechanisms

These diagrams help developers understand how ONNX runtime operates in production environments, enabling better optimization strategies and more robust application development.