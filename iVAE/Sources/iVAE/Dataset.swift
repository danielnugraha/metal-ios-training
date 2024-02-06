//
//  File.swift
//  
//
//  Created by Daniel Nugraha on 08.09.23.
//

import Foundation
import MetalPerformanceShaders

public struct Dataset {
    let totalNumberOfTrainImages: Int
    let trainImagePointer, trainLabelPointer: UnsafeMutablePointer<UInt8>
    let sizeTrainLabels, sizeTrainImages: Int
    let dataTrainImage: Data
    let dataTrainLabel: Data

    let totalNumberOfTestImages: Int;
    let testImagePointer, testLabelPointer: UnsafeMutablePointer<UInt8>
    let sizeTestLabels, sizeTestImages: Int
    let dataTestImage: Data
    let dataTestLabel: Data
    
    let imageSize: Int
    let imageMetadataSize: Int
    let labelMetadataSize: Int
    
    public init(totalNumberOfTrainImages: Int,
                trainImagePointer: UnsafeMutablePointer<UInt8>,
                trainLabelPointer: UnsafeMutablePointer<UInt8>,
                sizeTrainLabels: Int,
                sizeTrainImages: Int,
                dataTrainImage: Data,
                dataTrainLabel: Data,
                totalNumberOfTestImages: Int,
                testImagePointer: UnsafeMutablePointer<UInt8>,
                testLabelPointer: UnsafeMutablePointer<UInt8>,
                sizeTestLabels: Int,
                sizeTestImages: Int,
                dataTestImage: Data,
                dataTestLabel: Data,
                imageSize: Int,
                imageMetadataSize: Int,
                labelMetadataSize: Int) {
        self.totalNumberOfTrainImages = totalNumberOfTrainImages
        self.trainImagePointer = trainImagePointer
        self.trainLabelPointer = trainLabelPointer
        self.sizeTrainLabels = sizeTrainLabels
        self.sizeTrainImages = sizeTrainImages
        self.dataTrainImage = dataTrainImage
        self.dataTrainLabel = dataTrainLabel
        self.totalNumberOfTestImages = totalNumberOfTestImages
        self.testImagePointer = testImagePointer
        self.testLabelPointer = testLabelPointer
        self.sizeTestLabels = sizeTestLabels
        self.sizeTestImages = sizeTestImages
        self.dataTestImage = dataTestImage
        self.dataTestLabel = dataTestLabel
        self.imageSize = imageSize
        self.imageMetadataSize = imageMetadataSize
        self.labelMetadataSize = labelMetadataSize
    }
    
    func getRandomTrainingBatch(device: MTLDevice, batchSize: Int, lossStateBatch: UnsafeMutablePointer<[MPSCNNLossLabels]>) -> [MPSImage] {
        var trainBatch: [MPSImage] = []

        for _ in 0..<batchSize {
            let randomNormVal = Float(arc4random()) / Float(UINT32_MAX)
            let randomImageIdx = Int(randomNormVal * Float(totalNumberOfTrainImages))
                
            let trainImageDesc = MPSImageDescriptor(
                channelFormat: .unorm8,
                width: imageSize,
                height: imageSize,
                featureChannels: 1,
                numberOfImages: 1,
                usage: [.shaderWrite, .shaderRead]
            )
            
            let trainImage = MPSImage(device: device, imageDescriptor: trainImageDesc)
            let imageDataOffset = imageMetadataSize + randomImageIdx * imageSize * imageSize * MemoryLayout<UInt8>.size
    
            trainImage.writeBytes(UnsafeMutableRawPointer(trainImagePointer.advanced(by: imageDataOffset)), dataLayout: .HeightxWidthxFeatureChannels, imageIndex: 0)
            trainBatch.append(trainImage)
                
                
            let labelStart = trainLabelPointer.advanced(by: labelMetadataSize + randomImageIdx)
            var labelsBuffer = [Float](repeating: 0, count: 12)
            labelsBuffer[Int(labelStart.pointee)] = 1
            let labelsData = Data(bytes: labelsBuffer, count: 12 * MemoryLayout<Float>.size)
            guard let labelsDescriptor = MPSCNNLossDataDescriptor(data: labelsData, layout: .HeightxWidthxFeatureChannels, size: MTLSize(width: 1, height: 1, depth: 12)) else {
                return []
            }
            let lossState = MPSCNNLossLabels(device: device, labelsDescriptor: labelsDescriptor)
            lossStateBatch.pointee.append(lossState)
        }
        return trainBatch
    }
}
