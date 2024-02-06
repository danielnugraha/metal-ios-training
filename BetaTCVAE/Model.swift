//
//  Model.swift
//  BetaTCVAE
//
//  Created by Daniel Nugraha on 16.06.23.
//

import Foundation
import MetalPerformanceShaders
import Gzip
import iVAE

let MNISTImageSize = 28
let MNISTImageMetadataPrefixSize = 16
let MNISTLabelsMetadataPrefixSize = 8

func downloadMNIST(urlString: String) async -> Data? {
    if let url = URL(string: urlString) {
        do {
            let (fileURL, response) = try await URLSession.shared.download(from: url)
            let compresssedData = try Data(contentsOf: fileURL)
            print(response)
            return try compresssedData.gunzipped()
        } catch {
            print(error)
        }
    }
    return nil
}

public func createMNISTDataset() async -> Dataset? {
    let trainImagesURL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    let trainLabelsURL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    let testImagesURL = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    let testLabelsURL = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    
    let MNISTImageSize = 28
    let MNISTImageMetadataPrefixSize = 16
    let MNISTLabelsMetadataPrefixSize = 8
        
    var dataTrainImage: Data?
    var dataTrainLabel: Data?
    var dataTestImage: Data?
    var dataTestLabel: Data?
    
    if let imageTrainPath = Bundle.main.path(forResource: "train-images-idx3-ubyte", ofType: "data") {
        dataTrainImage = try? Data(contentsOf: URL(fileURLWithPath: imageTrainPath))
    } else {
        dataTrainImage = await downloadMNIST(urlString: trainImagesURL)
    }
            
    if let labelTrainPath = Bundle.main.path(forResource: "train-labels-idx1-ubyte", ofType: "data") {
        dataTrainLabel = try? Data(contentsOf: URL(fileURLWithPath: labelTrainPath))
    } else {
        dataTrainLabel = await downloadMNIST(urlString: trainLabelsURL)
    }
            
    if let imageTestPath = Bundle.main.path(forResource: "t10k-images-idx3-ubyte", ofType: "data") {
        dataTestImage = try? Data(contentsOf: URL(fileURLWithPath: imageTestPath))
    } else {
        dataTestImage = await downloadMNIST(urlString: testImagesURL)
    }
            
    if let labelTestPath = Bundle.main.path(forResource: "t10k-labels-idx1-ubyte", ofType: "data") {
        dataTestLabel = try? Data(contentsOf: URL(fileURLWithPath: labelTestPath))
    } else {
        dataTestLabel = await downloadMNIST(urlString: testLabelsURL)
    }
            
    let sizeTrainLabels = dataTrainLabel?.count ?? 0
    let sizeTrainImages = dataTrainImage?.count ?? 0
    let totalNumberOfTrainImages = sizeTrainLabels - MNISTLabelsMetadataPrefixSize
            
    let sizeTestLabels = dataTestLabel?.count ?? 0
    let sizeTestImages = dataTestImage?.count ?? 0
    let totalNumberOfTestImages = sizeTestLabels - MNISTLabelsMetadataPrefixSize
    
    
    let trainImagePointer = dataTrainImage?.withUnsafeMutableBytes {
        $0.baseAddress?.assumingMemoryBound(to: UInt8.self)
    }
    let trainLabelPointer = dataTrainLabel?.withUnsafeMutableBytes {
        $0.baseAddress?.assumingMemoryBound(to: UInt8.self)
    }
    let testImagePointer = dataTestImage?.withUnsafeMutableBytes {
        $0.baseAddress?.assumingMemoryBound(to: UInt8.self)
    }
    let testLabelPointer = dataTestLabel?.withUnsafeMutableBytes {
        $0.baseAddress?.assumingMemoryBound(to: UInt8.self)
    }
        
    return Dataset(totalNumberOfTrainImages: totalNumberOfTrainImages,
                   trainImagePointer: trainImagePointer!,
                   trainLabelPointer: trainLabelPointer!,
                   sizeTrainLabels: sizeTrainLabels,
                   sizeTrainImages: sizeTrainImages,
                   dataTrainImage: dataTrainImage!,
                   dataTrainLabel: dataTrainLabel!,
                   totalNumberOfTestImages: totalNumberOfTestImages,
                   testImagePointer: testImagePointer!,
                   testLabelPointer: testLabelPointer!,
                   sizeTestLabels: sizeTestLabels,
                   sizeTestImages: sizeTestImages,
                   dataTestImage: dataTestImage!,
                   dataTestLabel: dataTestLabel!,
                   imageSize: MNISTImageSize,
                   imageMetadataSize: MNISTImageMetadataPrefixSize,
                   labelMetadataSize: MNISTLabelsMetadataPrefixSize)
}

public func createConvMSEAE() -> [Layer] {
    [
        Convolution(kernelSize: .init(width: 3, height: 3), input: 1, output: 32, stride: 1, padding: .same),
        ReLU(),
        Pooling(mode: .max, filterSize: 2, stride: 2, padding: .same),
        Convolution(kernelSize: .init(width: 3, height: 3), input: 32, output: 32, stride: 1, padding: .same),
        ReLU(),
        Pooling(mode: .max, filterSize: 2, stride: 2, padding: .same),
        
        //Encoder
        ConvolutionTranspose(kernelSize: .init(width: 3, height: 3), input: 32, output: 32, stride: 2, padding: .same),
        ReLU(),
        ConvolutionTranspose(kernelSize: .init(width: 3, height: 3), input: 32, output: 32, stride: 2, padding: .same),
        ReLU(),
        Convolution(kernelSize: .init(width: 3, height: 3), input: 32, output: 1, stride: 1, padding: .same),
        Sigmoid()
    ]
}

public class AEModel: ObservableObject {
    public func runAutoencoder() async {
        if let dataset = await createMNISTDataset() {
            let graph = createConvMSEAE()
            let network = Sequential(layers: graph)
            network.compile(optimizer: .Adam, loss: .MSE)
            network.train(dataset: dataset)
        }
    }
}
