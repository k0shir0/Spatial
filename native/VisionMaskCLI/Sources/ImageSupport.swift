import CoreGraphics
import CoreVideo
import Darwin
import Foundation
import ImageIO
import UniformTypeIdentifiers

let supportedImageExtensions: Set<String> = [
    "heic", "heif", "jpeg", "jpg", "png", "tif", "tiff"
]

struct ImageMetadata {
    let url: URL
    let width: Int
    let height: Int
}

struct LoadedImage {
    let metadata: ImageMetadata
    let cgImage: CGImage
    let rgba: [UInt8]
}

struct GrayImage {
    let width: Int
    let height: Int
    let pixels: [UInt8]
    let bitsPerComponent: Int
    let bitsPerPixel: Int
    let sourceType: String?
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
    guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory), isDirectory.boolValue else {
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

    var stems: [String: URL] = [:]
    for file in files {
        let stem = file.deletingPathExtension().lastPathComponent.lowercased()
        if let previous = stems[stem] {
            throw CLIError.invalidInput(
                "Input files \(previous.lastPathComponent) and \(file.lastPathComponent) share a filename stem; both would map to \(stem).png."
            )
        }
        stems[stem] = file
    }
    return files
}

private func imageSource(for url: URL) throws -> CGImageSource {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
        throw CLIError.invalidInput("Could not open image: \(url.path)")
    }
    guard CGImageSourceGetCount(source) > 0 else {
        throw CLIError.invalidInput("Image contains no decodable frames: \(url.path)")
    }
    return source
}

func loadImageMetadata(_ url: URL) throws -> ImageMetadata {
    let source = try imageSource(for: url)
    let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
    let orientation = (properties?[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
    guard orientation == 1 else {
        throw CLIError.invalidInput(
            "\(url.lastPathComponent) has EXIF orientation \(orientation). Normalize it to upright pixels before masking so the PNG aligns exactly."
        )
    }

    guard
        let width = (properties?[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
        let height = (properties?[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue,
        width > 0,
        height > 0
    else {
        throw CLIError.invalidInput("Could not read pixel dimensions from \(url.path).")
    }
    return ImageMetadata(url: url, width: width, height: height)
}

func loadRGBAImage(_ url: URL) throws -> LoadedImage {
    let metadata = try loadImageMetadata(url)
    let source = try imageSource(for: url)
    let decodeOptions: [CFString: Any] = [
        kCGImageSourceShouldCache: true,
        kCGImageSourceShouldCacheImmediately: true
    ]
    guard let image = CGImageSourceCreateImageAtIndex(source, 0, decodeOptions as CFDictionary) else {
        throw CLIError.invalidInput("Could not decode image pixels: \(url.path)")
    }
    guard image.width == metadata.width, image.height == metadata.height else {
        throw CLIError.invalidInput("Decoded dimensions changed unexpectedly for \(url.lastPathComponent).")
    }

    var pixels = [UInt8](repeating: 0, count: metadata.width * metadata.height * 4)
    let rendered = pixels.withUnsafeMutableBytes { bytes -> Bool in
        guard let baseAddress = bytes.baseAddress else { return false }
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue
            | CGImageAlphaInfo.premultipliedLast.rawValue
        guard let context = CGContext(
            data: baseAddress,
            width: metadata.width,
            height: metadata.height,
            bitsPerComponent: 8,
            bytesPerRow: metadata.width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: bitmapInfo
        ) else {
            return false
        }
        context.translateBy(x: 0, y: CGFloat(metadata.height))
        context.scaleBy(x: 1, y: -1)
        context.interpolationQuality = .none
        context.draw(image, in: CGRect(x: 0, y: 0, width: metadata.width, height: metadata.height))
        return true
    }
    guard rendered else {
        throw CLIError.processing("Could not render RGBA pixels for \(url.lastPathComponent).")
    }

    return LoadedImage(metadata: metadata, cgImage: image, rgba: pixels)
}

func loadGrayImage(_ url: URL) throws -> GrayImage {
    let source = try imageSource(for: url)
    guard let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw CLIError.validation("Could not decode mask: \(url.path)")
    }

    var pixels = [UInt8](repeating: 0, count: image.width * image.height)
    let rendered = pixels.withUnsafeMutableBytes { bytes -> Bool in
        guard let baseAddress = bytes.baseAddress else { return false }
        guard let context = CGContext(
            data: baseAddress,
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
        throw CLIError.validation("Could not render grayscale mask pixels: \(url.path)")
    }

    return GrayImage(
        width: image.width,
        height: image.height,
        pixels: pixels,
        bitsPerComponent: image.bitsPerComponent,
        bitsPerPixel: image.bitsPerPixel,
        sourceType: CGImageSourceGetType(source) as String?
    )
}

func writeGrayscalePNG(
    pixels: [UInt8],
    width: Int,
    height: Int,
    to url: URL,
    overwrite: Bool
) throws {
    guard pixels.count == width * height else {
        throw CLIError.processing("Internal mask byte count does not match \(width)x\(height).")
    }

    let fileManager = FileManager.default
    if fileManager.fileExists(atPath: url.path) {
        guard overwrite else {
            throw CLIError.invalidInput("Output already exists: \(url.path)")
        }
        try fileManager.removeItem(at: url)
    }

    let data = Data(pixels)
    guard let provider = CGDataProvider(data: data as CFData) else {
        throw CLIError.processing("Could not create a data provider for \(url.lastPathComponent).")
    }
    guard let image = CGImage(
        width: width,
        height: height,
        bitsPerComponent: 8,
        bitsPerPixel: 8,
        bytesPerRow: width,
        space: CGColorSpaceCreateDeviceGray(),
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
        provider: provider,
        decode: nil,
        shouldInterpolate: false,
        intent: .defaultIntent
    ) else {
        throw CLIError.processing("Could not construct the 8-bit mask image for \(url.lastPathComponent).")
    }
    guard let destination = CGImageDestinationCreateWithURL(
        url as CFURL,
        UTType.png.identifier as CFString,
        1,
        nil
    ) else {
        throw CLIError.processing("Could not create PNG destination: \(url.path)")
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
        throw CLIError.processing("Could not finalize PNG: \(url.path)")
    }
}

private func writeRGBApng(
    pixels: [UInt8],
    width: Int,
    height: Int,
    to url: URL,
    overwrite: Bool
) throws {
    guard pixels.count == width * height * 4 else {
        throw CLIError.processing("Internal RGBA byte count does not match \(width)x\(height).")
    }
    let fileManager = FileManager.default
    if fileManager.fileExists(atPath: url.path) {
        guard overwrite else { throw CLIError.invalidInput("Debug output already exists: \(url.path)") }
        try fileManager.removeItem(at: url)
    }

    let data = Data(pixels)
    guard let provider = CGDataProvider(data: data as CFData) else {
        throw CLIError.processing("Could not create debug-image data for \(url.lastPathComponent).")
    }
    let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
    guard let image = CGImage(
        width: width,
        height: height,
        bitsPerComponent: 8,
        bitsPerPixel: 32,
        bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
        provider: provider,
        decode: nil,
        shouldInterpolate: false,
        intent: .defaultIntent
    ), let destination = CGImageDestinationCreateWithURL(
        url as CFURL,
        UTType.png.identifier as CFString,
        1,
        nil
    ) else {
        throw CLIError.processing("Could not create debug PNG: \(url.path)")
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
        throw CLIError.processing("Could not finalize debug PNG: \(url.path)")
    }
}

func debugOutputURLs(for inputURL: URL, in debugDirectory: URL) -> (cutout: URL, overlay: URL) {
    let stem = inputURL.deletingPathExtension().lastPathComponent
    return (
        debugDirectory.appendingPathComponent("\(stem)_cutout.png"),
        debugDirectory.appendingPathComponent("\(stem)_overlay.png")
    )
}

func writeDebugReviewImages(
    image: LoadedImage,
    mask: [UInt8],
    directory: URL,
    overwrite: Bool
) throws -> (cutout: URL, overlay: URL) {
    let width = image.metadata.width
    let height = image.metadata.height
    guard mask.count == width * height else {
        throw CLIError.processing("Debug mask dimensions do not match source pixels.")
    }

    var cutout = image.rgba
    var overlay = image.rgba
    for index in mask.indices {
        let pixel = index * 4
        let inside = mask[index] != 0
        if inside {
            cutout[pixel + 3] = 255
        } else {
            // The CGImage is premultiplied RGBA, so transparent pixels must also
            // have zero color channels.
            cutout[pixel] = 0
            cutout[pixel + 1] = 0
            cutout[pixel + 2] = 0
            cutout[pixel + 3] = 0
        }

        if inside {
            overlay[pixel] = UInt8((Int(image.rgba[pixel]) * 3 + 25) / 4)
            overlay[pixel + 1] = UInt8((Int(image.rgba[pixel + 1]) * 3 + 255) / 4)
            overlay[pixel + 2] = UInt8((Int(image.rgba[pixel + 2]) * 3 + 45) / 4)
        } else {
            overlay[pixel] = UInt8(Int(image.rgba[pixel]) * 35 / 100)
            overlay[pixel + 1] = UInt8(Int(image.rgba[pixel + 1]) * 35 / 100)
            overlay[pixel + 2] = UInt8(Int(image.rgba[pixel + 2]) * 35 / 100)
        }
        overlay[pixel + 3] = 255
    }

    // A yellow one-pixel contour makes hand/object boundary mistakes easy to spot.
    if width > 2, height > 2 {
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let index = y * width + x
                let inside = mask[index] != 0
                if (mask[index - 1] != 0) != inside
                    || (mask[index + 1] != 0) != inside
                    || (mask[index - width] != 0) != inside
                    || (mask[index + width] != 0) != inside {
                    let pixel = index * 4
                    overlay[pixel] = 255
                    overlay[pixel + 1] = 210
                    overlay[pixel + 2] = 0
                }
            }
        }
    }

    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let urls = debugOutputURLs(for: image.metadata.url, in: directory)
    try writeRGBApng(pixels: cutout, width: width, height: height, to: urls.cutout, overwrite: overwrite)
    try writeRGBApng(pixels: overlay, width: width, height: height, to: urls.overlay, overwrite: overwrite)
    return urls
}

func outputURL(for inputURL: URL, in outputDirectory: URL) -> URL {
    outputDirectory
        .appendingPathComponent(inputURL.deletingPathExtension().lastPathComponent)
        .appendingPathExtension("png")
}
