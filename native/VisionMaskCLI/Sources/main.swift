import Darwin
import Foundation

private struct FrameReport: Codable {
    let source: String
    let mask: String
    let width: Int
    let height: Int
    let selectedInstance: Int
    let detectedInstances: Int
    let selectionScore: Double
    let foregroundFraction: Double
    let greenFraction: Double
    let removedFraction: Double
    let debugCutout: String?
    let debugOverlay: String?
}

private struct RunReport: Codable {
    let backend: String
    let selection: String
    let personThreshold: Float
    let occluderMarginPixels: Int
    let edgeErosionPixels: Int
    let frames: [FrameReport]
}

private func preflight(options: Options, images: [URL]) throws {
    for imageURL in images {
        _ = try loadImageMetadata(imageURL)
        let destination = outputURL(for: imageURL, in: options.outputDirectory)
        if FileManager.default.fileExists(atPath: destination.path), !options.overwrite {
            throw CLIError.invalidInput("Output already exists: \(destination.path). Pass --overwrite to replace it.")
        }
        if let debugDirectory = options.debugDirectory {
            let debugURLs = debugOutputURLs(for: imageURL, in: debugDirectory)
            for debugURL in [debugURLs.cutout, debugURLs.overlay]
                where FileManager.default.fileExists(atPath: debugURL.path) && !options.overwrite {
                throw CLIError.invalidInput("Debug output already exists: \(debugURL.path). Pass --overwrite to replace it.")
            }
        }
    }
    let reportURL = options.outputDirectory.appendingPathComponent("mask_report.json")
    if FileManager.default.fileExists(atPath: reportURL.path), !options.overwrite {
        throw CLIError.invalidInput("Output already exists: \(reportURL.path). Pass --overwrite to replace it.")
    }
}

private func run(_ options: Options) throws {
    let images = try discoverImages(in: options.inputDirectory, includePattern: options.includePattern)

    if options.validateOnly {
        try validateMaskDirectory(options: options, images: images)
        return
    }

    try preflight(options: options, images: images)
    if options.dryRun {
        print("Dry run: \(images.count) upright images passed decode and output preflight.")
        for imageURL in images {
            let metadata = try loadImageMetadata(imageURL)
            let destination = outputURL(for: imageURL, in: options.outputDirectory)
            print("  \(imageURL.lastPathComponent) [\(metadata.width)x\(metadata.height)] -> \(destination.lastPathComponent)")
            if let debugDirectory = options.debugDirectory {
                let debugURLs = debugOutputURLs(for: imageURL, in: debugDirectory)
                print("    review -> \(debugURLs.cutout.lastPathComponent), \(debugURLs.overlay.lastPathComponent)")
            }
        }
        print("No Vision requests ran and no files were written.")
        return
    }

    try FileManager.default.createDirectory(at: options.outputDirectory, withIntermediateDirectories: true)
    var reports: [FrameReport] = []
    reports.reserveCapacity(images.count)

    for (offset, imageURL) in images.enumerated() {
        let result: Result<(LoadedImage, GeneratedMask), Error> = autoreleasepool {
            Result {
                let image = try loadRGBAImage(imageURL)
                let mask = try generateMask(for: image, options: options)
                return (image, mask)
            }
        }
        let (image, generated) = try result.get()
        let destination = outputURL(for: imageURL, in: options.outputDirectory)
        try writeGrayscalePNG(
            pixels: generated.pixels,
            width: generated.width,
            height: generated.height,
            to: destination,
            overwrite: options.overwrite
        )
        _ = try validateMask(
            at: destination, for: image.metadata,
            maximumForeground: options.objectSeedX == nil ? 0.35 : 0.50
        )
        let debugURLs = try options.debugDirectory.map {
            try writeDebugReviewImages(
                image: image,
                mask: generated.pixels,
                directory: $0,
                overwrite: options.overwrite
            )
        }

        reports.append(FrameReport(
            source: imageURL.path,
            mask: destination.path,
            width: generated.width,
            height: generated.height,
            selectedInstance: generated.selection.identifier,
            detectedInstances: generated.selection.instanceCount,
            selectionScore: generated.selection.score,
            foregroundFraction: generated.foregroundFraction,
            greenFraction: generated.greenFraction,
            removedFraction: generated.removedFraction,
            debugCutout: debugURLs?.cutout.path,
            debugOverlay: debugURLs?.overlay.path
        ))
        print(String(
            format: "[%d/%d] %@ -> %@ | instance %d/%d, foreground %.2f%%, green %.2f%%, excluded %.2f%%",
            offset + 1,
            images.count,
            imageURL.lastPathComponent,
            destination.lastPathComponent,
            generated.selection.identifier,
            generated.selection.instanceCount,
            generated.foregroundFraction * 100,
            generated.greenFraction * 100,
            generated.removedFraction * 100
        ))
    }

    let report = RunReport(
        backend: "Apple Vision foreground-instance + person segmentation",
        selection: options.objectSeedX == nil
            ? "centered green instance; person/skin subtraction; fail closed"
            : "explicit normalized object seed; person/skin subtraction; fail closed",
        personThreshold: options.personThreshold,
        occluderMarginPixels: options.occluderMargin,
        edgeErosionPixels: options.edgeErosion,
        frames: reports
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let reportData = try encoder.encode(report) + Data([0x0a])
    try reportData.write(
        to: options.outputDirectory.appendingPathComponent("mask_report.json"),
        options: .atomic
    )
    print("Generated and validated \(reports.count) masks sequentially.")
}

do {
    let options = try Options.parse(Array(CommandLine.arguments.dropFirst()))
    try run(options)
} catch CLIError.help {
    print(Options.usage)
    Darwin.exit(EXIT_SUCCESS)
} catch {
    fputs("vision-mask: \(error.localizedDescription)\n", stderr)
    Darwin.exit(EXIT_FAILURE)
}
