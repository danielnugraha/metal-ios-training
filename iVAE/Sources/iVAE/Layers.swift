//
//  Typing.swift
//  BetaTCVAE
//
//  Created by Daniel Nugraha on 29.06.23.
//

import Foundation
import MetalPerformanceShaders

public protocol Layer: Codable {
    func createNode(source: MPSNNImageNode, device: MTLDevice, commandQueue: MTLCommandQueue, optimizer: Optimizer) -> MPSNNFilterNode
}

var num = 0

public enum Padding: Int, Codable {
    case same
    case valid
    
    var padding: MPSNNDefaultPadding {
        switch self {
        case .same:
            return MPSNNDefaultPadding(method: .sizeSame)
        case .valid:
            return MPSNNDefaultPadding(method: .validOnly)
        }
    }
}

public enum PoolingMode: Int, Codable {
    case max
    case average
}

public struct Convolution: Layer {
    let kernelSize: CGSize
    let inputFeatureChannels, outputFeatureChannels, stride: Int
    let padding: Padding
    
    public init(kernelSize: CGSize,
                input inputFeatureChannels: Int,
                output outputFeatureChannels: Int,
                stride: Int,
                padding: Padding) {
        self.kernelSize = kernelSize
        self.inputFeatureChannels = inputFeatureChannels
        self.outputFeatureChannels = outputFeatureChannels
        self.stride = stride
        self.padding = padding
    }
    
    public func createNode(source: MPSNNImageNode, device: MTLDevice, commandQueue: MTLCommandQueue, optimizer: Optimizer) -> MPSNNFilterNode {
        let dataSource: DataSource = .init(kernelWidth: Int(kernelSize.width),
                                           kernelHeight: Int(kernelSize.height),
                                           inputFeatureChannels: inputFeatureChannels,
                                           outputFeatureChannels: outputFeatureChannels,
                                           stride: stride,
                                           commandQueue: commandQueue,
                                           name: "Conv\(num+=1)",
                                           optimizer: optimizer)
        let convolution = MPSCNNConvolutionNode(source: source, weights: dataSource)
        convolution.paddingPolicy = self.padding.padding
        dataSource.getParameters()
        return convolution
    }
}

public struct ConvolutionTranspose: Layer {
    let kernelSize: CGSize
    let inputFeatureChannels, outputFeatureChannels, stride: Int
    let padding: Padding
    
    public init(kernelSize: CGSize,
                input inputFeatureChannels: Int,
                output outputFeatureChannels: Int,
                stride: Int,
                padding: Padding) {
        self.kernelSize = kernelSize
        self.inputFeatureChannels = inputFeatureChannels
        self.outputFeatureChannels = outputFeatureChannels
        self.stride = stride
        self.padding = padding
    }
    
    public func createNode(source: MPSNNImageNode, device: MTLDevice, commandQueue: MTLCommandQueue, optimizer: Optimizer) -> MPSNNFilterNode {
        let dataSource: DataSource = .init(kernelWidth: Int(kernelSize.width),
                                           kernelHeight: Int(kernelSize.height),
                                           inputFeatureChannels: inputFeatureChannels,
                                           outputFeatureChannels: outputFeatureChannels,
                                           stride: stride,
                                           commandQueue: commandQueue,
                                           name: "ConvT\(num+=1)",
                                           optimizer: optimizer)
        let convolutionTranspose = MPSCNNConvolutionTransposeNode(source: source, weights: dataSource)
        convolutionTranspose.paddingPolicy = self.padding.padding
        dataSource.getParameters()
        return convolutionTranspose
    }
}

public struct Dense: Layer {
    let kernelSize: CGSize
    let inputFeatureChannels, outputFeatureChannels, stride: Int
    
    public init(kernelSize: CGSize,
                input inputFeatureChannels: Int,
                output outputFeatureChannels: Int,
                stride: Int) {
        self.kernelSize = kernelSize
        self.inputFeatureChannels = inputFeatureChannels
        self.outputFeatureChannels = outputFeatureChannels
        self.stride = stride
    }
    
    public func createNode(source: MPSNNImageNode, device: MTLDevice, commandQueue: MTLCommandQueue, optimizer: Optimizer) -> MPSNNFilterNode {
        let dataSource: DataSource = .init(kernelWidth: Int(kernelSize.width),
                                           kernelHeight: Int(kernelSize.height),
                                           inputFeatureChannels: inputFeatureChannels,
                                           outputFeatureChannels: outputFeatureChannels,
                                           stride: stride,
                                           commandQueue: commandQueue,
                                           name: "Dense\(num+=1)",
                                           optimizer: optimizer)
        return MPSCNNFullyConnectedNode(source: source, weights: dataSource)
    }
}

public struct Pooling: Layer {
    let mode: PoolingMode
    let filterSize: Int
    let stride: Int
    let padding: Padding
    
    public init(mode: PoolingMode, filterSize: Int, stride: Int, padding: Padding) {
        self.mode = mode
        self.filterSize = filterSize
        self.stride = stride
        self.padding = padding
    }
    
    public func createNode(source: MPSNNImageNode, device: MTLDevice, commandQueue: MTLCommandQueue, optimizer: Optimizer) -> MPSNNFilterNode {
        switch mode {
        case .max:
            let node = MPSCNNPoolingMaxNode(source: source,
                                            filterSize: self.filterSize,
                                            stride: self.stride)
            node.paddingPolicy = self.padding.padding
            return node
        case .average:
            let node = MPSCNNPoolingAverageNode(source: source,
                                                filterSize: self.filterSize,
                                                stride: self.stride)
            node.paddingPolicy = self.padding.padding
            return node
        }
    }
}

public struct ReLU: Layer {
    public init() {}
    public func createNode(source: MPSNNImageNode, device: MTLDevice, commandQueue: MTLCommandQueue, optimizer: Optimizer) -> MPSNNFilterNode {
        MPSCNNNeuronReLUNode(source: source, a: 0.0)
    }
}

public struct Sigmoid: Layer {
    public init() {}
    public func createNode(source: MPSNNImageNode, device: MTLDevice, commandQueue: MTLCommandQueue, optimizer: Optimizer) -> MPSNNFilterNode {
        MPSCNNNeuronSigmoidNode(source: source)
    }
}

public struct Flatten: Layer {
    let width: Int
    
    public init(width: Int) {
        self.width = width
    }
    
    public func createNode(source: MPSNNImageNode, device: MTLDevice, commandQueue: MTLCommandQueue, optimizer: Optimizer) -> MPSNNFilterNode {
        MPSNNReshapeNode(source: source,
                         resultWidth: 1,
                         resultHeight: 1,
                         resultFeatureChannels: self.width)
    }
}

public struct Dropout: Layer {
    let keepProbability: Float
    
    public init(keepProbability: Float) {
        self.keepProbability = keepProbability
    }
    
    public func createNode(source: MPSNNImageNode, device: MTLDevice, commandQueue: MTLCommandQueue, optimizer: Optimizer) -> MPSNNFilterNode {
        MPSCNNDropoutNode(source: source, keepProbability: self.keepProbability)
    }
}
