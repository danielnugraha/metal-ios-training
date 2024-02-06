//
//  LossFunction.swift
//  BetaTCVAE
//
//  Created by Daniel Nugraha on 16.06.23.
//

import Foundation
import MetalPerformanceShaders

public struct MSE: Layer {
    public func createNode(source: MPSNNImageNode, device: MTLDevice, commandQueue: MTLCommandQueue, optimizer: Optimizer) -> MPSNNFilterNode {
        let lossDescriptor = MPSCNNLossDescriptor(type: .meanSquaredError, reductionType: .mean)
        return MPSCNNLossNode(source: source, lossDescriptor: lossDescriptor)
    }
}
