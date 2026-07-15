import CoreGraphics
import Darwin
import Foundation
import ImageIO

let supportedImageExtensions: Set<String> = [
    "heic", "heif", "jpeg", "jpg", "png", "tif", "tiff"
]

struct LoadedImage {
    let url: URL
    let width: Int
    let height: Int
    let cgImage: CGImage
    let rgba: [UInt8]
}

struct LoadedMask {
    let url: URL
    let width: Int
    let height: Int
    let pixels: [UInt8]
    let foregroundFraction: Double
}

struct PreparedImage {
    let cgImage: CGImage
    let maskPath: String?
    let foregroundFraction: Double?
    let cropXYWH: [Int]?
}

func wildcardMatch(_ pattern: String, _ filename: String) -> Bool {
    pattern.withCString { patternPointer in
        filename.withCString { filenamePointer in
            fnmatch(patternPointer, filenamePointer, 0) == 0
        }
    }
}

func discoverImages(in directory: URL, includePattern: String) throws -> [URL] {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory),
          isDirectory.boolValue else {
        throw CLIError.invalidInput("Input directory does not exist: \(directory.path)")
    }

    let contents = try FileManager.default.contentsOfDirectory(
        at: directory,
        includingPropertiesForKeys: [.isRegularFileKey],
        options: [.skipsHiddenFiles]
    )
    let files = try contents.filter { url in
        let values = try url.resourceValues(forKeys: [.isRegularFileKey])
        return values.isRegularFile == true
            && supportedImageExtensions.contains(url.pathExtension.lowercased())
            && wildcardMatch(includePattern, url.lastPathComponent)
    }.sorted {
        $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending
    }

    guard !files.isEmpty else {
        throw CLIError.invalidInput(
            "No supported images in \(directory.path) matched --include '\(includePattern)'."
        )
    }
    return files
}

func discoverMasks(in directory: URL) throws -> [String: URL] {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory),
          isDirectory.boolValue else {
        throw CLIError.invalidInput("Mask directory does not exist: \(directory.path)")
    }
    let contents = try FileManager.default.contentsOfDirectory(
        at: directory,
        includingPropertiesForKeys: [.isRegularFileKey],
        options: [.skipsHiddenFiles]
    )
    var result: [String: URL] = [:]
    for url in contents where url.pathExtension.lowercased() == "png" {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey])
        guard values.isRegularFile == true else { continue }
        let stem = url.deletingPathExtension().lastPathComponent.lowercased()
        guard result[stem] == nil else {
            throw CLIError.invalidInput("Multiple masks share filename stem '\(stem)'.")
        }
        result[stem] = url
    }
    guard !result.isEmpty else {
        throw CLIError.invalidInput("Mask directory contains no PNG files: \(directory.path)")
    }
    return result
}

private func imageSource(for url: URL) throws -> CGImageSource {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          CGImageSourceGetCount(source) > 0 else {
        throw CLIError.invalidInput("Could not open image: \(url.path)")
    }
    return source
}

private func decodedUprightImage(_ url: URL) throws -> CGImage {
    let source = try imageSource(for: url)
    let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
    let orientation = (properties?[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
    guard orientation == 1 else {
        throw CLIError.invalidInput(
            "\(url.lastPathComponent) has EXIF orientation \(orientation); normalize it to upright pixels first."
        )
    }
    let options: [CFString: Any] = [
        kCGImageSourceShouldCache: true,
        kCGImageSourceShouldCacheImmediately: true
    ]
    guard let image = CGImageSourceCreateImageAtIndex(source, 0, options as CFDictionary) else {
        throw CLIError.invalidInput("Could not decode image pixels: \(url.path)")
    }
    return image
}

func loadRGBAImage(_ url: URL) throws -> LoadedImage {
    let image = try decodedUprightImage(url)
    let width = image.width
    let height = image.height
    var pixels = [UInt8](repeating: 0, count: width * height * 4)
    let rendered = pixels.withUnsafeMutableBytes { bytes -> Bool in
        guard let address = bytes.baseAddress,
              let context = CGContext(
                data: address,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue
                    | CGImageAlphaInfo.premultipliedLast.rawValue
              ) else {
            return false
        }
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)
        context.interpolationQuality = .none
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return true
    }
    guard rendered else {
        throw CLIError.processing("Could not render pixels for \(url.lastPathComponent).")
    }
    return LoadedImage(url: url, width: width, height: height, cgImage: image, rgba: pixels)
}

func loadMask(_ url: URL, expectedWidth: Int, expectedHeight: Int) throws -> LoadedMask {
    let image = try decodedUprightImage(url)
    guard image.width == expectedWidth, image.height == expectedHeight else {
        throw CLIError.invalidInput(
            "Mask \(url.lastPathComponent) is \(image.width)x\(image.height), expected \(expectedWidth)x\(expectedHeight)."
        )
    }
    var pixels = [UInt8](repeating: 0, count: image.width * image.height)
    let rendered = pixels.withUnsafeMutableBytes { bytes -> Bool in
        guard let address = bytes.baseAddress,
              let context = CGContext(
                data: address,
                width: image.width,
                height: image.height,
                bitsPerComponent: 8,
                bytesPerRow: image.width,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue
              ) else {
            return false
        }
        context.translateBy(x: 0, y: CGFloat(image.height))
        context.scaleBy(x: 1, y: -1)
        context.interpolationQuality = .none
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return true
    }
    guard rendered else {
        throw CLIError.processing("Could not render mask pixels for \(url.lastPathComponent).")
    }

    var foreground = 0
    for value in pixels {
        guard value == 0 || value == 255 else {
            throw CLIError.invalidInput(
                "Mask \(url.lastPathComponent) is not binary 0/255 grayscale."
            )
        }
        if value != 0 { foreground += 1 }
    }
    let fraction = Double(foreground) / Double(max(1, pixels.count))
    guard fraction >= 0.001, fraction <= 0.95 else {
        throw CLIError.invalidInput(
            String(format: "Mask %@ foreground %.3f%% is outside 0.1%%...95%%.", url.lastPathComponent, fraction * 100)
        )
    }
    return LoadedMask(
        url: url,
        width: image.width,
        height: image.height,
        pixels: pixels,
        foregroundFraction: fraction
    )
}

private func rgbaImage(pixels: [UInt8], width: Int, height: Int) throws -> CGImage {
    let data = Data(pixels)
    guard let provider = CGDataProvider(data: data as CFData),
          let image = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(
                rawValue: CGBitmapInfo.byteOrder32Big.rawValue
                    | CGImageAlphaInfo.premultipliedLast.rawValue
            ),
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
          ) else {
        throw CLIError.processing("Could not create masked classification image.")
    }
    return image
}

func prepareImage(_ image: LoadedImage, mask: LoadedMask?, paddingFraction: Double) throws -> PreparedImage {
    guard let mask else {
        return PreparedImage(
            cgImage: image.cgImage,
            maskPath: nil,
            foregroundFraction: nil,
            cropXYWH: nil
        )
    }
    precondition(mask.width == image.width && mask.height == image.height)

    var minimumX = image.width
    var minimumY = image.height
    var maximumX = -1
    var maximumY = -1
    for index in mask.pixels.indices where mask.pixels[index] != 0 {
        let x = index % image.width
        let y = index / image.width
        minimumX = min(minimumX, x)
        minimumY = min(minimumY, y)
        maximumX = max(maximumX, x)
        maximumY = max(maximumY, y)
    }
    guard maximumX >= minimumX, maximumY >= minimumY else {
        throw CLIError.invalidInput("Mask \(mask.url.lastPathComponent) contains no foreground.")
    }

    let objectWidth = maximumX - minimumX + 1
    let objectHeight = maximumY - minimumY + 1
    let padding = Int((Double(max(objectWidth, objectHeight)) * paddingFraction).rounded())
    let x0 = max(0, minimumX - padding)
    let y0 = max(0, minimumY - padding)
    let x1 = min(image.width, maximumX + padding + 1)
    let y1 = min(image.height, maximumY + padding + 1)
    let cropWidth = x1 - x0
    let cropHeight = y1 - y0

    var crop = [UInt8](repeating: 0, count: cropWidth * cropHeight * 4)
    for y in 0..<cropHeight {
        for x in 0..<cropWidth {
            let sourceIndex = (y0 + y) * image.width + (x0 + x)
            let destinationIndex = (y * cropWidth + x) * 4
            if mask.pixels[sourceIndex] != 0 {
                let sourcePixel = sourceIndex * 4
                crop[destinationIndex] = image.rgba[sourcePixel]
                crop[destinationIndex + 1] = image.rgba[sourcePixel + 1]
                crop[destinationIndex + 2] = image.rgba[sourcePixel + 2]
            } else {
                // Mid-gray is deliberately neutral; it contributes no semantic texture.
                crop[destinationIndex] = 127
                crop[destinationIndex + 1] = 127
                crop[destinationIndex + 2] = 127
            }
            crop[destinationIndex + 3] = 255
        }
    }

    return PreparedImage(
        cgImage: try rgbaImage(pixels: crop, width: cropWidth, height: cropHeight),
        maskPath: mask.url.path,
        foregroundFraction: mask.foregroundFraction,
        cropXYWH: [x0, y0, cropWidth, cropHeight]
    )
}
