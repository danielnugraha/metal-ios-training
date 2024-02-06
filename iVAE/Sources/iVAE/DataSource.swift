//
//  DataSource.swift
//  BetaTCVAE
//
//  Created by Daniel Nugraha on 16.06.23.
//

import Foundation
import MetalPerformanceShaders

public let gDevice = MTLCreateSystemDefaultDevice()!

public class DataSource: NSObject, MPSCNNConvolutionDataSource, Codable {
    private var convDescriptor: MPSCNNConvolutionDescriptor
    private var name: String
    private var learningRate, epsilon: Float
    private var beta1, beta2: Double
    
    private var weightPointer: UnsafeMutableRawPointer
    private var biasPointer: UnsafeMutablePointer<Float>
    
    private let commandQueue: MTLCommandQueue
    
    private var weightMomentumVector,
                biasMomentumVector,
                weightVelocityVector,
                biasVelocityVector,
                weightVector,
                biasVector: MPSVector
    
    private var updater: MPSNNOptimizerAdam
    private var convWtsAndBias: MPSCNNConvolutionWeightsAndBiasesState
    
    private enum CodingKeys: String, CodingKey {
        case beta1
        case beta2
        case epsilon
        case name
        case kernelSize
        case inputFeatureChannels
        case outputFeatureChannels
        case weightVector
        case biasVector
        case weightMomentum
        case biasMomentum
        case weightVelocity
        case biasVelocity
        case learningRate
    }
        
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encode(beta1, forKey: .beta1)
        try container.encode(beta2, forKey: .beta2)
        try container.encode(epsilon, forKey: .epsilon)
        try container.encode(CGSize(width: convDescriptor.kernelWidth, height: convDescriptor.kernelHeight), forKey: .kernelSize)
        try container.encode(convDescriptor.inputFeatureChannels, forKey: .inputFeatureChannels)
        try container.encode(convDescriptor.outputFeatureChannels, forKey: .outputFeatureChannels)
        try container.encode(weightVector, forKey: .weightVector)
        try container.encode(biasVector, forKey: .biasVector)
        try container.encode(weightMomentumVector, forKey: .weightMomentum)
        try container.encode(biasMomentumVector, forKey: .biasMomentum)
        try container.encode(weightVelocityVector, forKey: .weightVelocity)
        try container.encode(biasVelocityVector, forKey: .biasVelocity)
        try container.encode(learningRate, forKey: .learningRate)
    }
        
    public required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
            
        let device = MTLCreateSystemDefaultDevice()!
        commandQueue = device.makeCommandQueue()!
            
        learningRate = try container.decode(Float.self, forKey: .learningRate)
        name = try container.decode(String.self, forKey: .name)
        beta1 = try container.decode(Double.self, forKey: .beta1)
        beta2 = try container.decode(Double.self, forKey: .beta2)
        epsilon = try container.decode(Float.self, forKey: .epsilon)
        let kernelSize = try container.decode(CGSize.self, forKey: .kernelSize)
        let inputFeatureChannels = try container.decode(Int.self, forKey: .inputFeatureChannels)
        let outputFeatureChannels = try container.decode(Int.self, forKey: .outputFeatureChannels)
        convDescriptor = .init(kernelWidth: Int(kernelSize.width), kernelHeight: Int(kernelSize.height), inputFeatureChannels: inputFeatureChannels, outputFeatureChannels: outputFeatureChannels)
        convDescriptor.fusedNeuronDescriptor = .cnnNeuronDescriptor(with: .none)
            
        weightVector = try container.decode(MPSVector.self, forKey: .weightVector)
        biasVector = try container.decode(MPSVector.self, forKey: .biasVector)
        weightMomentumVector = try container.decode(MPSVector.self, forKey: .weightMomentum)
        biasMomentumVector = try container.decode(MPSVector.self, forKey: .biasMomentum)
        weightVelocityVector = try container.decode(MPSVector.self, forKey: .weightVelocity)
        biasVelocityVector = try container.decode(MPSVector.self, forKey: .biasVelocity)
        weightPointer = weightVector.data.contents()
        biasPointer = biasVector.data.contents().assumingMemoryBound(to: Float.self)
            
        convWtsAndBias = .init(weights: weightVector.data, biases: biasVector.data)
            
        let optimizerDescriptor = MPSNNOptimizerDescriptor(learningRate: learningRate, gradientRescale: 1.0, regularizationType: .None, regularizationScale: 1.0)
        
        updater = MPSNNOptimizerAdam(device: device, beta1: beta1, beta2: beta2, epsilon: epsilon, timeStep: 0, optimizerDescriptor: optimizerDescriptor)
    }
    
    public init(kernelWidth: Int,
                kernelHeight: Int,
                inputFeatureChannels: Int,
                outputFeatureChannels: Int,
                stride: Int,
                commandQueue: MTLCommandQueue,
                name: String,
                optimizer: Optimizer = .Adam) {
        self.name = name
        self.commandQueue = commandQueue
        self.convDescriptor = MPSCNNConvolutionDescriptor(kernelWidth: kernelWidth, kernelHeight: kernelHeight, inputFeatureChannels: inputFeatureChannels, outputFeatureChannels: outputFeatureChannels)
        self.convDescriptor.strideInPixelsX = stride
        self.convDescriptor.strideInPixelsY = stride
        self.convDescriptor.fusedNeuronDescriptor = MPSNNNeuronDescriptor.cnnNeuronDescriptor(with: .none)

        let lenWeights = inputFeatureChannels * kernelHeight * kernelWidth * outputFeatureChannels
        
        let sizeWeights = lenWeights * MemoryLayout<Float32>.size
        let sizeBiases = outputFeatureChannels * MemoryLayout<Float32>.size
        
        self.learningRate = Float(0.00005);
        self.beta1 = 0.9;
        self.beta2 = 0.999;
        self.epsilon = Float(1e-08);
        
        switch optimizer {
        case .Adam:
            let descriptor = MPSNNOptimizerDescriptor(learningRate: learningRate,
                                                      gradientRescale: Float(1.0),
                                                      regularizationType: .None,
                                                      regularizationScale: Float(1.0))
            
            self.updater = MPSNNOptimizerAdam(device: gDevice,
                                              beta1: Double(beta1),
                                              beta2: Double(beta2),
                                              epsilon: epsilon,
                                              timeStep: 0,
                                              optimizerDescriptor: descriptor)
        }
        
        
        let vDescWeights = MPSVectorDescriptor(length: lenWeights, dataType: .float32)
        self.weightMomentumVector = MPSVector(device: gDevice, descriptor: vDescWeights)
        self.weightVelocityVector = MPSVector(device: gDevice, descriptor: vDescWeights)
        self.weightVector = MPSVector(device: gDevice, descriptor: vDescWeights)
        
        let vDescBiases = MPSVectorDescriptor(length: outputFeatureChannels, dataType: .float32)
        self.biasMomentumVector = MPSVector(device: gDevice, descriptor: vDescBiases)
        self.biasVelocityVector = MPSVector(device: gDevice, descriptor: vDescBiases)
        self.biasVector = MPSVector(device: gDevice, descriptor: vDescBiases)
        
        self.convWtsAndBias = .init(weights: weightVector.data, biases: biasVector.data)
        var zero = Float.zero, biasInit = Float.zero
        self.weightPointer = weightVector.data.contents()
        let weightVelocityPointer = weightVelocityVector.data.contents()
        let weightMomentumPointer = weightMomentumVector.data.contents()
        
        memset_pattern4(weightVelocityPointer, &zero, sizeWeights)
        memset_pattern4(weightMomentumPointer, &zero, sizeWeights)
                
        self.biasVector = MPSVector(device: gDevice, descriptor: vDescBiases)
        self.biasVelocityVector = MPSVector(device: gDevice, descriptor: vDescBiases)
        self.biasMomentumVector = MPSVector(device: gDevice, descriptor: vDescBiases)
        self.biasPointer = biasVector.data.contents().assumingMemoryBound(to: Float.self)
        let biasVelocityPointer = biasVelocityVector.data.contents()
        let biasMomentumPointer = biasMomentumVector.data.contents()
                
        memset_pattern4(biasPointer, &biasInit, sizeBiases)
        memset_pattern4(biasVelocityPointer, &zero, sizeBiases)
        memset_pattern4(biasMomentumPointer, &zero, sizeBiases)
        
        let commandBuffer = MPSCommandBuffer(from: commandQueue)
                
        let limit = sqrt(6.0 / Float(inputFeatureChannels + outputFeatureChannels))
        let randomDescriptor = MPSMatrixRandomDistributionDescriptor.uniformDistributionDescriptor(withMinimum: -limit, maximum: limit)
        let randomKernel = MPSMatrixRandomMTGP32(device: gDevice, destinationDataType: .float32, seed: 0, distributionDescriptor: randomDescriptor)
        randomKernel.encode(commandBuffer: commandBuffer, destinationVector: weightVector)
                
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    public func label() -> String? {
        name
    }
    
    public func copy(with zone: NSZone? = nil) -> Any {
        self
    }
    
    public func dataType() -> MPSDataType {
        .float32
    }
    
    public func descriptor() -> MPSCNNConvolutionDescriptor {
        convDescriptor
    }
    
    public func weights() -> UnsafeMutableRawPointer {
        weightPointer
    }
    
    public func biasTerms() -> UnsafeMutablePointer<Float>? {
        biasPointer
    }
    
    public func load() -> Bool {
        checkpointWithCommandQueue()
        return true
    }
    
    func checkpointWithCommandQueue(){
        autoreleasepool {
            let commandBuffer = MPSCommandBuffer(from: commandQueue)
            convWtsAndBias.synchronize(on: commandBuffer)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
    }
    
    public func purge() { }
    
    public func update(with commandBuffer: MTLCommandBuffer,
                       gradientState: MPSCNNConvolutionGradientState,
                       sourceState: MPSCNNConvolutionWeightsAndBiasesState) -> MPSCNNConvolutionWeightsAndBiasesState? {
        updater.encode(commandBuffer: commandBuffer,
                       convolutionGradientState: gradientState,
                       convolutionSourceState: sourceState,
                       inputMomentumVectors: [weightMomentumVector, biasMomentumVector],
                       inputVelocityVectors: [weightVelocityVector, biasVelocityVector],
                       resultState: convWtsAndBias)
        return convWtsAndBias
    }
    
    public func getParameters() {
        let lenWeights = convDescriptor.inputFeatureChannels * convDescriptor.kernelHeight * convDescriptor.kernelWidth * convDescriptor.outputFeatureChannels
        let floatPointer = weightPointer.bindMemory(to: Float32.self, capacity: lenWeights)
        let floatBuffer = UnsafeBufferPointer(start: floatPointer, count: lenWeights)
        print("Weights size: \(lenWeights)")
        print("Array count: \(Array(floatBuffer).count)")
        print("Array weights: \(Array(floatBuffer).suffix(10))")
    }
}

extension KeyedEncodingContainer {
    mutating func encode(_ value: MTLBuffer, forKey key: K) throws {
        let length = value.length
        let count = length / 4
        let result = value.contents().bindMemory(to: Float.self, capacity: count)
        var arr = Array(repeating: Float.zero, count: count)
        for i in 0..<count {
            arr[i] = result[i]
        }
        try encode(arr, forKey: key)
    }
    
    mutating func encode(_ value: MPSVector, forKey key: K) throws {
        try encode(value.data, forKey: key)
    }
}

extension KeyedDecodingContainer {
    func decode(_ type: MTLBuffer.Type, forKey key: K) throws -> MTLBuffer {
        let arr = try decode([Float].self, forKey: key)
        let length = arr.count * 4
        let device = MTLCreateSystemDefaultDevice()!
        let buffer = device.makeBuffer(bytes: arr, length: length, options: [])!
        return buffer
    }
    
    func decode(_ type: MPSVector.Type, forKey key: K) throws -> MPSVector {
        let buffer = try decode(MTLBuffer.self, forKey: key)
        let descriptor = MPSVectorDescriptor(length: buffer.length/4, dataType: .float32)
        return MPSVector(buffer: buffer, descriptor: descriptor)
    }
}
