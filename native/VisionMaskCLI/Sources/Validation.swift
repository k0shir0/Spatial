import Foundation
import UniformTypeIdentifiers

struct MaskStatistics {
    let foregroundFraction: Double
    let isBinary: Bool
}

func maskStatistics(_ pixels: [UInt8]) -> MaskStatistics {
    guard !pixels.isEmpty else { return MaskStatistics(foregroundFraction: 0, isBinary: false) }
    var foreground = 0
    var binary = true
    for value in pixels {
        if value != 0 { foreground += 1 }
        if value != 0 && value != 255 { binary = false }
    }
    return MaskStatistics(
        foregroundFraction: Double(foreground) / Double(pixels.count),
        isBinary: binary
    )
}

@discardableResult
func validateMask(
    at maskURL: URL,
    for metadata: ImageMetadata,
    minimumForeground: Double = 0.0015,
    maximumForeground: Double = 0.35
) throws -> MaskStatistics {
    let mask = try loadGrayImage(maskURL)
    guard mask.sourceType == UTType.png.identifier else {
        throw CLIError.validation("\(maskURL.lastPathComponent) is not encoded as PNG.")
    }
    guard mask.width == metadata.width, mask.height == metadata.height else {
        throw CLIError.validation(
            "\(maskURL.lastPathComponent) is \(mask.width)x\(mask.height), expected \(metadata.width)x\(metadata.height)."
        )
    }
    guard mask.bitsPerComponent == 8, mask.bitsPerPixel == 8 else {
        throw CLIError.validation(
            "\(maskURL.lastPathComponent) is not a one-channel 8-bit image (\(mask.bitsPerComponent) bits/component, \(mask.bitsPerPixel) bits/pixel)."
        )
    }

    let statistics = maskStatistics(mask.pixels)
    guard statistics.isBinary else {
        throw CLIError.validation("\(maskURL.lastPathComponent) contains values other than 0 and 255.")
    }
    guard statistics.foregroundFraction >= minimumForeground,
          statistics.foregroundFraction <= maximumForeground else {
        throw CLIError.validation(
            String(
                format: "%@ foreground coverage %.3f%% is outside %.3f%%...%.1f%%.",
                maskURL.lastPathComponent,
                statistics.foregroundFraction * 100,
                minimumForeground * 100,
                maximumForeground * 100
            )
        )
    }
    return statistics
}

func validateMaskDirectory(options: Options, images: [URL]) throws {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: options.outputDirectory.path, isDirectory: &isDirectory), isDirectory.boolValue else {
        throw CLIError.validation("Mask directory does not exist: \(options.outputDirectory.path)")
    }

    let expectedStems = Set(images.map { $0.deletingPathExtension().lastPathComponent.lowercased() })
    let maskFiles = try FileManager.default.contentsOfDirectory(
        at: options.outputDirectory,
        includingPropertiesForKeys: [.isRegularFileKey],
        options: [.skipsHiddenFiles]
    ).filter { $0.pathExtension.lowercased() == "png" }
    let actualStems = Set(maskFiles.map { $0.deletingPathExtension().lastPathComponent.lowercased() })

    let missing = expectedStems.subtracting(actualStems).sorted()
    let extra = actualStems.subtracting(expectedStems).sorted()
    guard missing.isEmpty else {
        throw CLIError.validation("Missing masks for filename stems: \(missing.joined(separator: ", ")).")
    }
    guard extra.isEmpty else {
        throw CLIError.validation("Unexpected PNG masks without matching inputs: \(extra.joined(separator: ", ")).")
    }

    var totalCoverage = 0.0
    for imageURL in images {
        let metadata = try loadImageMetadata(imageURL)
        let maskURL = outputURL(for: imageURL, in: options.outputDirectory)
        let statistics = try validateMask(
            at: maskURL, for: metadata,
            maximumForeground: options.objectSeedX == nil ? 0.35 : 0.50
        )
        totalCoverage += statistics.foregroundFraction
        print(String(format: "valid %-28@ %6.2f%% foreground", maskURL.lastPathComponent, statistics.foregroundFraction * 100))
    }
    print(String(format: "Validated %d masks; mean foreground %.2f%%.", images.count, totalCoverage / Double(images.count) * 100))
}
