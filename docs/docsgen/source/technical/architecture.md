<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX Technical Architecture

This document provides a comprehensive overview of the ONNX (Open Neural Network Exchange) technical architecture, including detailed diagrams that illustrate the system's components, data flows, and interactions.

## Overview

ONNX is an open ecosystem that empowers AI developers to choose the right tools as their project evolves. It provides an open source format for AI models and defines an extensible computation graph model, built-in operators, and standard data types.

## System Architecture

The following diagram shows the high-level architecture of the ONNX ecosystem:

```mermaid
graph TB
    %% Training Frameworks
    subgraph TF["Training Frameworks"]
        PyTorch["PyTorch"]
        TensorFlow["TensorFlow"] 
        Keras["Keras"]
        Scikit["Scikit-learn"]
        Others["Other ML Frameworks"]
    end

    %% ONNX Core
    subgraph ONNX["ONNX Core System"]
        subgraph IR["Intermediate Representation"]
            ModelProto["ModelProto"]
            GraphProto["GraphProto"]
            NodeProto["NodeProto"]
            ValueInfo["ValueInfoProto"]
        end
        
        subgraph OPS["Operator System"]
            OpSchema["Operator Schemas"]
            OpSets["Operator Sets"]
            TypeSystem["Type System"]
            ShapeInf["Shape Inference"]
        end
        
        subgraph TOOLS["Core Tools"]
            Checker["Model Checker"]
            Optimizer["Graph Optimizer"]
            Converter["Version Converter"]
            ShapeInfTool["Shape Inference Tool"]
        end
    end

    %% Runtime Backends
    subgraph RT["Runtime Backends"]
        ONNXRT["ONNX Runtime"]
        TensorRT["TensorRT"]
        OpenVINO["Intel OpenVINO"]
        CoreML["Apple Core ML"]
        RKNN["Rockchip RKNN"]
        CustomRT["Custom Runtimes"]
    end

    %% Hardware Targets
    subgraph HW["Hardware Targets"]
        CPU["CPU"]
        GPU["GPU"]
        NPU["NPU"]
        FPGA["FPGA"]
        Mobile["Mobile Devices"]
        Edge["Edge Devices"]
    end

    %% Connections
    TF --> ONNX
    ONNX --> RT
    RT --> HW

    %% Styling
    classDef framework fill:#e1f5fe
    classDef onnx fill:#f3e5f5
    classDef runtime fill:#e8f5e8
    classDef hardware fill:#fff3e0

    class PyTorch,TensorFlow,Keras,Scikit,Others framework
    class ModelProto,GraphProto,NodeProto,ValueInfo,OpSchema,OpSets,TypeSystem,ShapeInf,Checker,Optimizer,Converter,ShapeInfTool onnx
    class ONNXRT,TensorRT,OpenVINO,CoreML,RKNN,CustomRT runtime
    class CPU,GPU,NPU,FPGA,Mobile,Edge hardware
```

## Model Representation Architecture

ONNX models are represented using Protocol Buffers. The following diagram shows the hierarchical structure:

```mermaid
graph TD
    Model["ModelProto<br/>Top-level container"]
    
    Model --> MetaData["Model Metadata<br/>- IR Version<br/>- Producer Name<br/>- Model Version"]
    Model --> Graph["GraphProto<br/>Main computation graph"]
    Model --> OpsetImports["Opset Imports<br/>Required operator sets"]
    
    Graph --> Nodes["NodeProto[]<br/>Computation nodes"]
    Graph --> Inputs["ValueInfoProto[]<br/>Graph inputs"]
    Graph --> Outputs["ValueInfoProto[]<br/>Graph outputs"]
    Graph --> Initializers["TensorProto[]<br/>Model parameters"]
    Graph --> ValueInfos["ValueInfoProto[]<br/>Intermediate values"]
    
    Nodes --> NodeDetails["NodeProto Details<br/>- Operation type<br/>- Attributes<br/>- Input names<br/>- Output names<br/>- Domain"]
    
    Inputs --> TypeInfo["Type Information<br/>- Tensor type<br/>- Shape info<br/>- Element type"]
    Outputs --> TypeInfo
    ValueInfos --> TypeInfo
    
    Initializers --> TensorData["Tensor Data<br/>- Raw data<br/>- Data type<br/>- Dimensions<br/>- External data"]

    %% Styling
    classDef model fill:#e3f2fd
    classDef graph fill:#f1f8e9
    classDef node fill:#fff3e0
    classDef data fill:#fce4ec
    
    class Model model
    class Graph,Inputs,Outputs,ValueInfos graph
    class Nodes,NodeDetails node
    class MetaData,OpsetImports,Initializers,TensorData,TypeInfo data
```

## Operator System Architecture

The ONNX operator system provides a flexible and extensible framework for defining operations:

```mermaid
graph TB
    subgraph OpSystem["Operator System"]
        subgraph Domains["Operator Domains"]
            DefaultDomain["Default Domain<br/>(ai.onnx)"]
            MLDomain["ML Domain<br/>(ai.onnx.ml)"]
            TrainingDomain["Training Domain<br/>(ai.onnx.training)"]
            CustomDomains["Custom Domains"]
        end
        
        subgraph OpSets["Operator Sets"]
            OpSet1["Opset 1"]
            OpSet18["Opset 18"]
            OpSetN["Opset N"]
            MLOpSet["ML Opset"]
        end
        
        subgraph Schemas["Operator Schemas"]
            OpSchema["OpSchema<br/>- Name & Domain<br/>- Since Version<br/>- Attributes<br/>- Inputs/Outputs<br/>- Type Constraints"]
            TypeConstraints["Type Constraints<br/>- Allowed types<br/>- Type relationships"]
            ShapeInference["Shape Inference<br/>- Input shapes<br/>- Output shapes<br/>- Broadcasting rules"]
        end
        
        subgraph TypeSys["Type System"]
            PrimitiveTypes["Primitive Types<br/>- float16/32/64<br/>- int8/16/32/64<br/>- bool, string"]
            TensorTypes["Tensor Types<br/>- Dense tensors<br/>- Sparse tensors"]
            SequenceTypes["Sequence Types"]
            MapTypes["Map Types"]
            OptionalTypes["Optional Types"]
        end
    end
    
    subgraph Processing["Processing Pipeline"]
        Registration["Schema Registration"]
        Validation["Type & Shape Validation"]
        Inference["Shape & Type Inference"]
        Optimization["Graph Optimization"]
    end
    
    Domains --> OpSets
    OpSets --> Schemas
    Schemas --> TypeSys
    Schemas --> Processing
    
    %% Styling
    classDef domain fill:#e8f5e8
    classDef opset fill:#e3f2fd
    classDef schema fill:#fff3e0
    classDef type fill:#f3e5f5
    classDef process fill:#fce4ec
    
    class DefaultDomain,MLDomain,TrainingDomain,CustomDomains domain
    class OpSet1,OpSet18,OpSetN,MLOpSet opset
    class OpSchema,TypeConstraints,ShapeInference schema
    class PrimitiveTypes,TensorTypes,SequenceTypes,MapTypes,OptionalTypes type
    class Registration,Validation,Inference,Optimization process
```

## Data Flow Architecture

This diagram illustrates how data flows through an ONNX model during inference:

```mermaid
flowchart TD
    Input["Model Inputs<br/>External data"]
    
    subgraph ModelExec["Model Execution"]
        Initializers["Initializers<br/>Model parameters"]
        
        subgraph ComputeGraph["Computation Graph"]
            Node1["Node 1<br/>Operation"]
            Node2["Node 2<br/>Operation"]
            Node3["Node 3<br/>Operation"]
            NodeN["Node N<br/>Operation"]
            
            Node1 --> Node2
            Node2 --> Node3
            Node3 --> NodeN
        end
        
        subgraph DataFlow["Data Flow"]
            Tensor1["Intermediate<br/>Tensor 1"]
            Tensor2["Intermediate<br/>Tensor 2"]
            Tensor3["Intermediate<br/>Tensor 3"]
        end
        
        Node1 --> Tensor1
        Node2 --> Tensor2
        Node3 --> Tensor3
        Tensor1 --> Node2
        Tensor2 --> Node3
        Tensor3 --> NodeN
    end
    
    Output["Model Outputs<br/>Results"]
    
    Input --> Node1
    Initializers --> Node1
    Initializers --> Node2
    Initializers --> Node3
    NodeN --> Output
    
    %% Additional processing
    subgraph Runtime["Runtime Processing"]
        TypeCheck["Type Checking"]
        ShapeInf["Shape Inference"]
        MemoryMgmt["Memory Management"]
        Optimization["Graph Optimization"]
    end
    
    Input --> Runtime
    Runtime --> ComputeGraph
    
    %% Styling
    classDef input fill:#e8f5e8
    classDef compute fill:#e3f2fd
    classDef data fill:#fff3e0
    classDef output fill:#f3e5f5
    classDef runtime fill:#fce4ec
    
    class Input input
    class Node1,Node2,Node3,NodeN,ComputeGraph compute
    class Tensor1,Tensor2,Tensor3,Initializers,DataFlow data
    class Output output
    class TypeCheck,ShapeInf,MemoryMgmt,Optimization,Runtime runtime
```

## Shape Inference Architecture

Shape inference is a critical component that determines tensor shapes at graph compilation time:

```mermaid
graph TD
    subgraph ShapeInfSystem["Shape Inference System"]
        GraphInput["Graph with<br/>Unknown Shapes"]
        
        subgraph Analysis["Shape Analysis"]
            StaticShapes["Static Shape<br/>Analysis"]
            SymbolicShapes["Symbolic Shape<br/>Propagation"]
            PartialData["Partial Data<br/>Propagation"]
        end
        
        subgraph InferenceEngine["Inference Engine"]
            NodeLevel["Node-level<br/>Inference"]
            GraphLevel["Graph-level<br/>Inference"]
            TypeInference["Type Inference"]
        end
        
        subgraph ShapeRules["Shape Rules"]
            BroadcastRules["Broadcasting Rules"]
            ElementwiseRules["Element-wise Rules"]
            ReductionRules["Reduction Rules"]
            ReshapeRules["Reshape Rules"]
        end
        
        InferredGraph["Graph with<br/>Inferred Shapes"]
    end
    
    GraphInput --> Analysis
    Analysis --> InferenceEngine
    InferenceEngine --> ShapeRules
    ShapeRules --> InferredGraph
    
    subgraph SymbolSystem["Symbol System"]
        SymbolGen["Symbol Generation"]
        SymbolProp["Symbol Propagation"]
        SymbolMerge["Symbol Merging"]
    end
    
    SymbolicShapes --> SymbolSystem
    SymbolSystem --> GraphLevel
    
    %% Styling
    classDef input fill:#e8f5e8
    classDef analysis fill:#e3f2fd
    classDef engine fill:#fff3e0
    classDef rules fill:#f3e5f5
    classDef output fill:#fce4ec
    classDef symbol fill:#ede7f6
    
    class GraphInput input
    class StaticShapes,SymbolicShapes,PartialData,Analysis analysis
    class NodeLevel,GraphLevel,TypeInference,InferenceEngine engine
    class BroadcastRules,ElementwiseRules,ReductionRules,ReshapeRules,ShapeRules rules
    class InferredGraph output
    class SymbolGen,SymbolProp,SymbolMerge,SymbolSystem symbol
```

## Backend Integration Architecture

This diagram shows how ONNX integrates with various runtime backends:

```mermaid
graph TB
    ONNXModel["ONNX Model<br/>.onnx file"]
    
    subgraph BackendLayer["Backend Integration Layer"]
        ModelLoader["Model Loader"]
        GraphOpt["Graph Optimizer"]
        BackendAPI["Backend API"]
    end
    
    subgraph Backends["Runtime Backends"]
        subgraph ONNXRuntime["ONNX Runtime"]
            CPUProvider["CPU Execution Provider"]
            CUDAProvider["CUDA Execution Provider"]
            TensorRTProvider["TensorRT Execution Provider"]
            DirectMLProvider["DirectML Execution Provider"]
        end
        
        subgraph SpecializedBackends["Specialized Backends"]
            TensorRT["NVIDIA TensorRT"]
            OpenVINO["Intel OpenVINO"]
            CoreML["Apple Core ML"]
            RKNN["Rockchip RKNN"]
        end
        
        subgraph CustomBackends["Custom Backends"]
            CustomCPU["Custom CPU Backend"]
            CustomGPU["Custom GPU Backend"]
            CustomNPU["Custom NPU Backend"]
        end
    end
    
    subgraph HardwareLayer["Hardware Layer"]
        Intel["Intel CPUs"]
        NVIDIA["NVIDIA GPUs"]
        AMD["AMD GPUs"]
        ARM["ARM Processors"]
        AppleM["Apple Silicon"]
        Qualcomm["Qualcomm NPUs"]
    end
    
    ONNXModel --> BackendLayer
    BackendLayer --> Backends
    
    CPUProvider --> Intel
    CUDAProvider --> NVIDIA
    TensorRTProvider --> NVIDIA
    DirectMLProvider --> AMD
    
    TensorRT --> NVIDIA
    OpenVINO --> Intel
    CoreML --> AppleM
    RKNN --> ARM
    
    CustomCPU --> Intel
    CustomGPU --> NVIDIA
    CustomNPU --> Qualcomm
    
    %% Styling
    classDef model fill:#e8f5e8
    classDef backend fill:#e3f2fd
    classDef runtime fill:#fff3e0
    classDef specialized fill:#f3e5f5
    classDef custom fill:#fce4ec
    classDef hardware fill:#ede7f6
    
    class ONNXModel model
    class ModelLoader,GraphOpt,BackendAPI,BackendLayer backend
    class CPUProvider,CUDAProvider,TensorRTProvider,DirectMLProvider,ONNXRuntime runtime
    class TensorRT,OpenVINO,CoreML,RKNN,SpecializedBackends specialized
    class CustomCPU,CustomGPU,CustomNPU,CustomBackends custom
    class Intel,NVIDIA,AMD,ARM,AppleM,Qualcomm,HardwareLayer hardware
```

## Model Validation and Checking Architecture

ONNX provides comprehensive validation to ensure model correctness:

```mermaid
flowchart TD
    Model["ONNX Model"]
    
    subgraph ValidationSystem["Validation System"]
        subgraph SyntaxCheck["Syntax Validation"]
            ProtoValidation["Protobuf Schema<br/>Validation"]
            StructuralCheck["Structural<br/>Validation"]
        end
        
        subgraph SemanticCheck["Semantic Validation"]
            TypeCheck["Type Checking"]
            ShapeCheck["Shape Validation"]
            OpValidation["Operator Validation"]
            DomainCheck["Domain Validation"]
        end
        
        subgraph GraphCheck["Graph Validation"]
            TopologyCheck["Topology Check"]
            CycleDetection["Cycle Detection"]
            ConnectivityCheck["Connectivity Check"]
            InitializerCheck["Initializer Validation"]
        end
        
        subgraph ComplianceCheck["Compliance Validation"]
            OpsetCompliance["Opset Compliance"]
            VersionCheck["Version Compatibility"]
            StandardCompliance["ONNX Standard<br/>Compliance"]
        end
    end
    
    subgraph Results["Validation Results"]
        Valid["Valid Model<br/>✓ Passed all checks"]
        Invalid["Invalid Model<br/>✗ Validation errors"]
        Warnings["Valid with Warnings<br/>⚠ Minor issues"]
    end
    
    Model --> ValidationSystem
    
    SyntaxCheck --> SemanticCheck
    SemanticCheck --> GraphCheck
    GraphCheck --> ComplianceCheck
    
    ValidationSystem --> Valid
    ValidationSystem --> Invalid
    ValidationSystem --> Warnings
    
    %% Styling
    classDef model fill:#e8f5e8
    classDef syntax fill:#e3f2fd
    classDef semantic fill:#fff3e0
    classDef graph fill:#f3e5f5
    classDef compliance fill:#fce4ec
    classDef valid fill:#c8e6c9
    classDef invalid fill:#ffcdd2
    classDef warning fill:#fff9c4
    
    class Model model
    class ProtoValidation,StructuralCheck,SyntaxCheck syntax
    class TypeCheck,ShapeCheck,OpValidation,DomainCheck,SemanticCheck semantic
    class TopologyCheck,CycleDetection,ConnectivityCheck,InitializerCheck,GraphCheck graph
    class OpsetCompliance,VersionCheck,StandardCompliance,ComplianceCheck compliance
    class Valid valid
    class Invalid invalid
    class Warnings warning
```

## Version Management Architecture

ONNX supports versioning at multiple levels to ensure backward compatibility:

```mermaid
graph TB
    subgraph VersioningSystem["ONNX Versioning System"]
        subgraph IRVersioning["IR Versioning"]
            IRv1["IR Version 1<br/>Basic graph model"]
            IRv7["IR Version 7<br/>Training support"]
            IRv10["IR Version 10<br/>Current stable"]
        end
        
        subgraph OpsetVersioning["Opset Versioning"]
            DefaultOpset["Default Opset<br/>(ai.onnx)"]
            MLOpset["ML Opset<br/>(ai.onnx.ml)"]
            TrainingOpset["Training Opset<br/>(ai.onnx.training)"]
            
            DefaultOpset --> DefV1["Version 1"]
            DefaultOpset --> DefV18["Version 18"]
            DefaultOpset --> DefVN["Version N"]
            
            MLOpset --> MLV1["Version 1"]
            MLOpset --> MLV3["Version 3"]
            
            TrainingOpset --> TrainV1["Version 1"]
        end
        
        subgraph VersionConversion["Version Conversion"]
            Converter["ONNX Version<br/>Converter"]
            UpgradeRules["Upgrade Rules"]
            DowngradeRules["Downgrade Rules"]
            CompatCheck["Compatibility<br/>Checker"]
        end
        
        subgraph VersionPolicy["Versioning Policy"]
            BackwardCompat["Backward<br/>Compatibility"]
            ForwardCompat["Forward<br/>Compatibility"]
            DeprecationPolicy["Deprecation<br/>Policy"]
        end
    end
    
    IRVersioning --> OpsetVersioning
    OpsetVersioning --> VersionConversion
    VersionConversion --> VersionPolicy
    
    %% Model flow
    OldModel["Legacy ONNX Model<br/>Older version"]
    NewModel["Modern ONNX Model<br/>Newer version"]
    
    OldModel --> Converter
    Converter --> NewModel
    
    %% Styling
    classDef ir fill:#e8f5e8
    classDef opset fill:#e3f2fd
    classDef conversion fill:#fff3e0
    classDef policy fill:#f3e5f5
    classDef model fill:#fce4ec
    
    class IRv1,IRv7,IRv10,IRVersioning ir
    class DefaultOpset,MLOpset,TrainingOpset,DefV1,DefV18,DefVN,MLV1,MLV3,TrainV1,OpsetVersioning opset
    class Converter,UpgradeRules,DowngradeRules,CompatCheck,VersionConversion conversion
    class BackwardCompat,ForwardCompat,DeprecationPolicy,VersionPolicy policy
    class OldModel,NewModel model
```

## Summary

This technical architecture documentation provides a comprehensive view of the ONNX ecosystem, covering:

1. **System Architecture**: High-level view of frameworks, ONNX core, runtimes, and hardware
2. **Model Representation**: Detailed structure of ONNX models using Protocol Buffers
3. **Operator System**: Extensible operator framework with domains, opsets, and schemas
4. **Data Flow**: How data moves through computation graphs during inference
5. **Shape Inference**: Critical system for determining tensor shapes at compile time
6. **Backend Integration**: How ONNX works with various runtime backends and hardware
7. **Model Validation**: Comprehensive checking system for model correctness
8. **Version Management**: Multi-level versioning for backward and forward compatibility

Each diagram uses Mermaid syntax for clear visualization and can be rendered in any Markdown viewer that supports Mermaid diagrams. The architecture is designed to be modular, extensible, and performant across a wide range of AI/ML use cases and deployment scenarios.