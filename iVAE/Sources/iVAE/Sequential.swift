//
//  Sequential.swift
//  BetaTCVAE
//
//  Created by Daniel Nugraha on 01.09.23.
//

import Foundation
import MetalPerformanceShaders

public enum Optimizer {
    case Adam
}

public enum LossLayer {
    case MSE
}

public class Sequential {
    let device: MTLDevice?
    let commandQueue: MTLCommandQueue?
    var epochs: Int = 1
    var batchSize: Int = 40
    var inferenceGraph: MPSNNGraph?
    var trainingGraph: MPSNNGraph?
    var layers: [Layer] = []
    
    public init() {
        self.device = MTLCreateSystemDefaultDevice()
        self.commandQueue = device?.makeCommandQueue()
    }
    
    public convenience init(layers: [Layer]) {
        self.init()
        self.layers = layers
    }
    
    public func add(layer: Layer) {
        layers.append(layer)
    }
    
    public func compile(optimizer: Optimizer, loss: LossLayer) {
        if let device = self.device, let commandQueue = commandQueue {
            let trainingNodes = createTrainingGraph(device: device, commandQueue: commandQueue, optimizer: optimizer, loss: loss)
            let lossExitPoints = trainingNodes.trainingGraph(withSourceGradient: nil) { gradientNode, inferenceNode, inferenceSource, gradientSource in
                gradientNode.resultImage.format = .float32
            }
            
            let trainingGraph = MPSNNGraph(device: device, resultImage: lossExitPoints![0].resultImage, resultImageIsNeeded: true)
            trainingGraph?.format = .float32
            self.trainingGraph = trainingGraph
            
            let inferenceNodes = createInferenceGraph(device: device, commandQueue: commandQueue, optimizer: optimizer)
            let inferenceGraph = MPSNNGraph(device: device, resultImage: inferenceNodes.resultImage, resultImageIsNeeded: true)
            inferenceGraph?.format = .float32
            self.inferenceGraph = inferenceGraph
            
        }
    }
    
    private func createTrainingGraph(device: MTLDevice, commandQueue: MTLCommandQueue, optimizer: Optimizer, loss: LossLayer) -> MPSNNFilterNode {
        var source = MPSNNImageNode(handle: nil)
        var graph: MPSNNFilterNode
        for layer in layers {
            graph = layer.createNode(source: source, device: device, commandQueue: commandQueue, optimizer: optimizer)
            source = graph.resultImage
            source.format = .float32
        }
        
        switch loss {
        case .MSE:
            return MSE().createNode(source: source, device: device, commandQueue: commandQueue, optimizer: optimizer)
        }
    }
    
    private func createInferenceGraph(device: MTLDevice, commandQueue: MTLCommandQueue, optimizer: Optimizer) -> MPSNNFilterNode {
        var source = MPSNNImageNode(handle: nil)
        var graph: MPSNNFilterNode
        for layer in layers {
            graph = layer.createNode(source: source, device: device, commandQueue: commandQueue, optimizer: optimizer)
            source = graph.resultImage
            source.format = .float32
        }
        return MPSCNNSoftMaxNode(source: source)
    }
    
    func getOutputSize(dataset: Dataset) {
        if let device = self.device, let commandBuffer = commandQueue?.makeCommandBuffer() {
            var lossStateBatch: [MPSCNNLossLabels] = []
            let inputBatch = dataset.getRandomTrainingBatch(device: device, batchSize: 1, lossStateBatch: &lossStateBatch)
            let outputBatch = encodeInferenceBatchToCommandBuffer(commandBuffer: commandBuffer, sourceImages: inputBatch)
            
            MPSImageBatchSynchronize(outputBatch, commandBuffer)
            
            let nsOutput = NSArray(array: outputBatch)
            
            commandBuffer.addCompletedHandler() { _ in
                nsOutput.enumerateObjects() { outputImage, idx, stop in
                    if let outputImage = outputImage as? MPSImage {
                        print("Output size: width - \(outputImage.width), height - \(outputImage.height), featureChannels - \(outputImage.featureChannels)")
                    }
                }
            }
            
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
    }
    
    
    func encodeTrainingBatchToCommandBuffer(commandBuffer: MTLCommandBuffer,
                                            sourceImages: [MPSImage],
                                            lossStates: [MPSCNNLossLabels]) -> [MPSImage] {
            
        guard let returnImage = trainingGraph?.encodeBatch(to: commandBuffer,
                                                          sourceImages: [sourceImages],
                                                          sourceStates: [lossStates],
                                                          intermediateImages: nil,
                                                          destinationStates: nil) else {
            print("Unable to encode training batch to command buffer.")
            return []
        }

        MPSImageBatchSynchronize(returnImage, commandBuffer)

        return returnImage
    }
    
        
        
    func encodeInferenceBatchToCommandBuffer(commandBuffer: MTLCommandBuffer,
                                             sourceImages: [MPSImage]) -> [MPSImage] {
        guard let returnImage = inferenceGraph?.encodeBatch(to: commandBuffer,
                                                           sourceImages: [sourceImages],
                                                           sourceStates: nil,
                                                           intermediateImages: nil,
                                                           destinationStates: nil) else {
            print("Unable to encode inference batch to command buffer.")
            return []
        }
        return returnImage
    }
    
    private let doubleBufferingSemaphore = DispatchSemaphore(value: 2)
    
    public func train(dataset: Dataset, epoch: Int = 1, batchSize: Int = 40) {
        let trainingEpoch = 300
            
        for i in 0..<trainingEpoch {
            _ = trainIteration(iteration: i, numberOfIterations: trainingEpoch, dataset: dataset)
        }
    }
    
    public func trainIteration(iteration: Int,
                        numberOfIterations: Int,
                        dataset: Dataset) -> MTLCommandBuffer? {
        guard let device = device, let commandQueue = commandQueue else {
            print("Unable to get MTLDevice and CommandQueue.")
            return nil
        }
        let startTime = Date.now
        
        doubleBufferingSemaphore.wait()
            
        var lossStateBatch: [MPSCNNLossLabels] = []
        
        
        let commandBuffer = MPSCommandBuffer(from: commandQueue)
            
        let randomTrainBatch = dataset.getRandomTrainingBatch(device: device,
                                                        batchSize: batchSize,
                                                        lossStateBatch: &lossStateBatch)
            
        let returnBatch = encodeTrainingBatchToCommandBuffer(commandBuffer: commandBuffer,
                                                             sourceImages: randomTrainBatch,
                                                             lossStates: lossStateBatch)
            
        var outputBatch = [MPSImage]()
            
        for i in 0..<batchSize {
            outputBatch.append(lossStateBatch[i].lossImage())
        }
            
        commandBuffer.addCompletedHandler() { commandBuffer in
            self.doubleBufferingSemaphore.signal()
            
            let timeDiff = Date.now.timeIntervalSince(startTime)
            let trainingLoss = self.lossReduceSumAcrossBatch(batch: outputBatch)
            print(" Iteration \(iteration+1)/\(numberOfIterations), \(timeDiff)s, 0.Training loss = \(trainingLoss)\r", terminator: "")
            fflush(stdout)
                
            let err = commandBuffer.error
            if err != nil {
                print(err!.localizedDescription)
            }
        }
        
        MPSImageBatchSynchronize(returnBatch, commandBuffer);
        MPSImageBatchSynchronize(outputBatch, commandBuffer);
            
        commandBuffer.commit()
            
        return commandBuffer
    }
    
    private func lossReduceSumAcrossBatch(batch: [MPSImage]) -> Float {
        var ret = Float.zero
        for i in 0..<batch.count {
            var val = [Float.zero]
            batch[i].readBytes(&val, dataLayout: .HeightxWidthxFeatureChannels, imageIndex: 0)
            ret += Float(val[0]) / Float(batch.count)
        }
        return ret
     }
        
    
    public func train(trainSet: Dataset, evaluationSet: Dataset) {
        for i in 0..<epochs {
            autoreleasepool(invoking: {
                print("Starting epoch \(i)")
                trainEpoch(dataset: trainSet).waitUntilCompleted()
            })
        }
    }
    
    private func trainEpoch(dataset: Dataset) -> MTLCommandBuffer {
        let iterations = 0
        var latestCommandBuffer: MTLCommandBuffer?
        for i in 0..<iterations {
            latestCommandBuffer = trainIteration(iteration: i, numberOfIterations: iterations, dataset: dataset)
        }
        return latestCommandBuffer!
    }
    
    public func evaluateTestSet(dataset: Dataset, iterations: Int) {
        // Reset accuracy counters, begin a fresh test set evaluation.
        var gDone = 0
        var gCorrect = 0

        // Update inference graph weights with the trained weights.
        inferenceGraph?.reloadFromDataSources()

        // Create an input MPSImageBatch.
        let inputDesc = MPSImageDescriptor(channelFormat: .unorm8, width: dataset.imageSize, height: dataset.imageSize, featureChannels: 1, numberOfImages: 1, usage: .shaderRead)
        
        var lastCommandBuffer: MPSCommandBuffer?
        
        var inputBatch = [MPSImage]()
        for _ in 0..<batchSize {
            let inputImage = MPSImage(device: gDevice, imageDescriptor: inputDesc)
            inputBatch.append(inputImage)
        }
        
        guard let commandQueue = commandQueue else {
            print("Unable to get CommandQueue.")
            return
        }
        
        // Encoding each image.
        for currImageIdx in 0..<dataset.totalNumberOfTestImages {
            // Make an MPSCommandBuffer, when passed to the encode of MPSNNGraph, commitAndContinue will be automatically used.
            let commandBuffer = MPSCommandBuffer(from: commandQueue)

            // Write the image data to from MNIST Testing dataset to the MPSImageBatch.
            inputBatch.enumerated().forEach { idx, inputImage in
                let start = dataset.testImagePointer.advanced(by: dataset.imageMetadataSize + (dataset.imageSize * dataset.imageSize * currImageIdx + idx))
                inputImage.writeBytes(start, dataLayout: .HeightxWidthxFeatureChannels, imageIndex: 0)
            }

            // Encode inference network
            let outputBatch = encodeInferenceBatchToCommandBuffer(commandBuffer: commandBuffer, sourceImages: inputBatch)

            // Transfer data from GPU to CPU (will be a no-op on embedded GPUs)
            MPSImageBatchSynchronize(outputBatch, commandBuffer)

            commandBuffer.addCompletedHandler { _ in
                // Release double buffering semaphore for the next training iteration to be encoded.
                self.doubleBufferingSemaphore.signal()

                // Check the output of inference network to calculate accuracy.
                outputBatch.enumerated().forEach { idx, outputImage in
                    let labelStart = dataset.testLabelPointer.advanced(by: dataset.labelMetadataSize + currImageIdx + idx)
                    let image = outputImage
                    let size = image.width * image.height * image.featureChannels
                    
                    var vals = Array(repeating: Float32(-22), count: size)
                    var index = -1, maxV = Float32(-100)
                    
                    image.readBytes(&vals, dataLayout: .featureChannelsxHeightxWidth, imageIndex: 0)
                    
                    for i in 0..<image.featureChannels {
                        for j in 0..<image.height {
                            for k in 0..<image.width {
                                let val = vals[(i*image.height + j) * image.width + k]
                                if val > maxV {
                                    maxV = val
                                    index = (i*image.height + j) * image.width + k
                                    
                                }
                            }
                        }
                    }
                    if index == labelStart[0] {
                        gCorrect += 1
                    }
                    gDone += 1
                }
            }
            
            commandBuffer.commit()
            lastCommandBuffer = commandBuffer
        }

        // Wait for the last batch to be processed.
        
        lastCommandBuffer?.waitUntilCompleted()
        print("Test Set Accuracy = %f %%", (Float(gCorrect) / (Float(dataset.totalNumberOfTestImages) / Float(100))))
    }
}
