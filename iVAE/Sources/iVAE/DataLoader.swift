//
//  DataLoader.swift
//  BetaTCVAE
//
//  Created by Daniel Nugraha on 16.06.23.
//

import Foundation
import MetalPerformanceShaders
import CoreImage
import Accelerate

enum DataType: Int, Codable {
    case image = 0
    case bytes
}

class DataSample: Codable {
    var image: CGImage?
    var texture: MTLTexture?
    var bytes: [UInt8]?
    var type: DataType
    var label: Int
    
    func getMPSImage(device: MTLDevice) -> MPSImage {
        if type == .image {
            return MPSImage(texture: texture!, featureChannels: 1)
        } else {
            let bytes = bytes!
            let descriptor = MPSImageDescriptor(channelFormat: .unorm8, width: 1, height: 1, featureChannels: bytes.count, numberOfImages: 1, usage: [.shaderRead, .shaderWrite])
            let image = MPSImage(device: device, imageDescriptor: descriptor)
            image.writeBytes(bytes, dataLayout: .HeightxWidthxFeatureChannels, imageIndex: 0)
            return image
        }
    }
    
    private enum CodingKeys: String, CodingKey {
        case image
        case label
        case type
        case bytes
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(label, forKey: .label)
        try container.encode(type, forKey: .type)
        if type == .image {
            try container.encode(image?.png!.base64EncodedString(options: []), forKey: .image)
        } else {
            try container.encode(bytes!, forKey: .bytes)
        }
    }
    
    required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        label = try container.decode(Int.self, forKey: .label)
        type = try container.decode(DataType.self, forKey: .type)
        
        if type == .image {
            let base64Encoded = try container.decode(String.self, forKey: .image)
            let data = NSData(base64Encoded: base64Encoded, options: [])!
            
            self.image = CGImage(pngDataProviderSource: CGDataProvider(data: data)!, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
            self.texture = image?.texture
        } else {
            self.bytes = try container.decode([UInt8].self, forKey: .bytes)
        }
    }
    
    init(image: CGImage, label: Int) {
        self.type = .image
        self.label = label
        self.image = image
        self.texture = image.texture
    }
    
    init(bytes: [UInt8], label: Int) {
        self.type = .bytes
        self.bytes = bytes
        self.label = label
    }
}

extension CGImage {
    
    var texture: MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: width, height: height, mipmapped: false)
        
        let device = MTLCreateSystemDefaultDevice()!
        
        let texture = device.makeTexture(descriptor: descriptor)!
        let region = MTLRegion(origin: .init(x: 0, y: 0, z: 0), size: .init(width: width, height: height, depth: 1))
        
        var format = vImage_CGImageFormat(bitsPerComponent: UInt32(8), bitsPerPixel: UInt32(8), colorSpace: Unmanaged.passRetained(CGColorSpace(name: CGColorSpace.linearGray)!), bitmapInfo: .init(rawValue: CGImageAlphaInfo.none.rawValue), version: 0, decode: nil, renderingIntent: .defaultIntent)
        do {
            var sourceBuffer = try vImage_Buffer(cgImage: self, format: format)
            var error = vImage_Error()
            let destImage = vImageCreateCGImageFromBuffer(&sourceBuffer, &format, nil, nil, numericCast(kvImageNoFlags), &error).takeRetainedValue()
            
            guard error == noErr else {
                fatalError()
            }
            
            let dstData = destImage.dataProvider?.data
            let pixelData = CFDataGetBytePtr(dstData!)
            
            texture.replace(region: region, mipmapLevel: 0, withBytes: pixelData!, bytesPerRow: bytesPerRow)
            
            return texture
        } catch {
            fatalError(error.localizedDescription)
        }
    }
    
    func grayscale(size: CGSize) -> CGImage {
        let imageRect:CGRect = CGRect(origin: .zero, size: size)
        let colorSpace = CGColorSpace(name: CGColorSpace.linearGray)!
        let width = size.width
        let height = size.height
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        let context = CGContext(data: nil, width: Int(width), height: Int(height), bitsPerComponent: 8, bytesPerRow: 0, space: colorSpace, bitmapInfo: bitmapInfo.rawValue)
        context?.draw(self, in: imageRect)
        if let makeImg = context?.makeImage() {
            return makeImg
        }
        return self
    }
    
    var png: NSData? {
        guard let mutableData = CFDataCreateMutable(nil, 0),
              let destination = CGImageDestinationCreateWithData(mutableData, "public.png" as CFString, 1, nil) else { print("Unable to get PNG data"); return nil }
        CGImageDestinationAddImage(destination, self, nil)
        guard CGImageDestinationFinalize(destination) else { print("Unable to get PNG data"); return nil }
        return mutableData as NSData
    }
}

extension CIImage {
    var convertedCGImage: CGImage? {
        let context = CIContext(options: nil)
        return context.createCGImage(self, from: self.extent)
    }
    
    var inverted: CIImage {
        let filter = CIFilter(name: "CIColorInvert")!
        filter.setValue(self, forKey: kCIInputImageKey)
        
        return filter.outputImage ?? self
    }
    
    func resize(targetSize: CGSize) -> CIImage {
        let resizeFilter = CIFilter(name:"CILanczosScaleTransform")!

        let scale = targetSize.height / self.extent.height
        let aspectRatio = targetSize.width/(self.extent.width * scale)

        resizeFilter.setValue(self, forKey: kCIInputImageKey)
        resizeFilter.setValue(scale, forKey: kCIInputScaleKey)
        resizeFilter.setValue(aspectRatio, forKey: kCIInputAspectRatioKey)
        return resizeFilter.outputImage ?? self
    }
}
