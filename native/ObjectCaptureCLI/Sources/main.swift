import CoreGraphics
import CoreImage
import CoreVideo
import Darwin
import Foundation
import ImageIO
import RealityKit

private enum CLIError: LocalizedError {
    case usage(String)
    case invalidInput(String)
    case unsupported
    case reconstruction(String)

    var errorDescription: String? {
        switch self {
        case .usage(let message), .invalidInput(let message), .reconstruction(let message):
            return message
        case .unsupported:
            return "RealityKit Object Capture is unavailable on this Mac. Run this binary directly from Terminal on a supported Apple-silicon Mac."
        }
    }
}

private struct Options {
    let inputDirectory: URL
    let maskDirectory: URL?
    let outputURL: URL
    let detail: PhotogrammetrySession.Request.Detail
    let sequential: Bool
    let highSensitivity: Bool
    let objectMaskingEnabled: Bool
    let force: Bool

    static let usage = """
    Usage:
      object-capture --input <keyframe-dir> --output <model.usdz> [options]

    Options:
      --masks <mask-dir>          Grayscale PNG masks matched by filename stem.
                                  Black pixels are ignored; nonblack pixels are object.
      --detail <level>            preview, reduced, medium, full, or raw (default: reduced).
      --unordered                 Do not tell RealityKit that adjacent filenames are adjacent views.
      --normal-sensitivity        Use normal rather than high feature sensitivity.
      --no-object-masking         Disable both supplied and automatic object masking.
      --force                     Replace an existing output file.
      -h, --help                  Show this help.

    Every supplied mask must have the same pixel dimensions and filename stem as its image:
      keyframes/frame_0001.jpg  <->  masks/frame_0001.png
    """

    static func parse(_ arguments: [String]) throws -> Options {
        var input: String?
        var masks: String?
        var output: String?
        var detailName = "reduced"
        var sequential = true
        var highSensitivity = true
        var objectMaskingEnabled = true
        var force = false

        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            func nextValue() throws -> String {
                guard index + 1 < arguments.count else {
                    throw CLIError.usage("Missing value after \(argument).\n\n\(usage)")
                }
                index += 1
                return arguments[index]
            }

            switch argument {
            case "--input":
                input = try nextValue()
            case "--masks":
                masks = try nextValue()
            case "--output":
                output = try nextValue()
            case "--detail":
                detailName = try nextValue().lowercased()
            case "--unordered":
                sequential = false
            case "--normal-sensitivity":
                highSensitivity = false
            case "--no-object-masking":
                objectMaskingEnabled = false
            case "--force":
                force = true
            case "-h", "--help":
                print(usage)
                Darwin.exit(EXIT_SUCCESS)
            default:
                throw CLIError.usage("Unknown argument: \(argument)\n\n\(usage)")
            }
            index += 1
        }

        guard let input, let output else {
            throw CLIError.usage("Both --input and --output are required.\n\n\(usage)")
        }
        if masks != nil && !objectMaskingEnabled {
            throw CLIError.usage("--masks and --no-object-masking cannot be used together.")
        }

        let detail: PhotogrammetrySession.Request.Detail
        switch detailName {
        case "preview": detail = .preview
        case "reduced": detail = .reduced
        case "medium": detail = .medium
        case "full": detail = .full
        case "raw": detail = .raw
        default:
            throw CLIError.usage("Unknown detail level '\(detailName)'. Use preview, reduced, medium, full, or raw.")
        }

        return Options(
            inputDirectory: URL(fileURLWithPath: input, isDirectory: true).standardizedFileURL,
            maskDirectory: masks.map { URL(fileURLWithPath: $0, isDirectory: true).standardizedFileURL },
            outputURL: URL(fileURLWithPath: output).standardizedFileURL,
            detail: detail,
            sequential: sequential,
            highSensitivity: highSensitivity,
            objectMaskingEnabled: objectMaskingEnabled,
            force: force
        )
    }
}

private let supportedImageExtensions: Set<String> = [
    "heic", "heif", "jpeg", "jpg", "png", "tif", "tiff"
]

private func imageFiles(in directory: URL) throws -> [URL] {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory), isDirectory.boolValue else {
        throw CLIError.invalidInput("Input directory does not exist: \(directory.path)")
    }

    let contents = try FileManager.default.contentsOfDirectory(
        at: directory,
        includingPropertiesForKeys: [.isRegularFileKey],
        options: [.skipsHiddenFiles]
    )
    return contents
        .filter { supportedImageExtensions.contains($0.pathExtension.lowercased()) }
        .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
}

private func maskIndex(in directory: URL) throws -> [String: URL] {
    let files = try imageFiles(in: directory)
    var result: [String: URL] = [:]
    for file in files {
        let stem = file.deletingPathExtension().lastPathComponent
        if result[stem] != nil {
            throw CLIError.invalidInput("More than one mask has filename stem '\(stem)' in \(directory.path).")
        }
        result[stem] = file
    }
    return result
}

private func loadMask(
    from url: URL,
    expectedWidth: Int,
    expectedHeight: Int,
    context: CIContext
) throws -> CVPixelBuffer {
    guard
        let source = CGImageSourceCreateWithURL(url as CFURL, nil),
        let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
        throw CLIError.invalidInput("Could not decode mask: \(url.path)")
    }

    guard image.width == expectedWidth, image.height == expectedHeight else {
        throw CLIError.invalidInput(
            "Mask dimensions do not match its keyframe: \(url.lastPathComponent) is \(image.width)x\(image.height), expected \(expectedWidth)x\(expectedHeight)."
        )
    }

    let attributes: [CFString: Any] = [
        kCVPixelBufferCGImageCompatibilityKey: true,
        kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        kCVPixelBufferMetalCompatibilityKey: true
    ]
    var optionalBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        expectedWidth,
        expectedHeight,
        kCVPixelFormatType_OneComponent8,
        attributes as CFDictionary,
        &optionalBuffer
    )
    guard status == kCVReturnSuccess, let buffer = optionalBuffer else {
        throw CLIError.invalidInput("Could not allocate an 8-bit mask buffer for \(url.lastPathComponent) (CoreVideo status \(status)).")
    }

    let imageBounds = CGRect(x: 0, y: 0, width: expectedWidth, height: expectedHeight)
    context.render(CIImage(cgImage: image), to: buffer, bounds: imageBounds, colorSpace: CGColorSpaceCreateDeviceGray())
    return buffer
}

private func makeMaskedSamples(images: [URL], maskDirectory: URL) async throws -> [PhotogrammetrySample] {
    let masks = try maskIndex(in: maskDirectory)
    let context = CIContext(options: [.cacheIntermediates: false])
    var samples: [PhotogrammetrySample] = []
    samples.reserveCapacity(images.count)

    for imageURL in images {
        let stem = imageURL.deletingPathExtension().lastPathComponent
        guard let maskURL = masks[stem] else {
            throw CLIError.invalidInput("No mask matching keyframe '\(imageURL.lastPathComponent)' was found in \(maskDirectory.path).")
        }

        var sample = try await PhotogrammetrySample(contentsOf: imageURL)
        sample.objectMask = try loadMask(
            from: maskURL,
            expectedWidth: CVPixelBufferGetWidth(sample.image),
            expectedHeight: CVPixelBufferGetHeight(sample.image),
            context: context
        )
        samples.append(sample)
    }
    return samples
}

private func formatDuration(_ seconds: TimeInterval?) -> String {
    guard let seconds, seconds.isFinite else { return "unknown" }
    if seconds < 60 { return "\(Int(seconds.rounded()))s" }
    return "\(Int(seconds / 60))m \(Int(seconds.truncatingRemainder(dividingBy: 60).rounded()))s"
}

private func run(_ options: Options) async throws {
    guard PhotogrammetrySession.isSupported else { throw CLIError.unsupported }

    let images = try imageFiles(in: options.inputDirectory)
    guard images.count >= 3 else {
        throw CLIError.invalidInput("Object Capture needs at least 3 usable keyframes; found \(images.count) in \(options.inputDirectory.path).")
    }

    let fileManager = FileManager.default
    if fileManager.fileExists(atPath: options.outputURL.path) {
        guard options.force else {
            throw CLIError.invalidInput("Output already exists: \(options.outputURL.path). Pass --force to replace it.")
        }
        try fileManager.removeItem(at: options.outputURL)
    }
    try fileManager.createDirectory(
        at: options.outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )

    var configuration = PhotogrammetrySession.Configuration()
    configuration.sampleOrdering = options.sequential ? .sequential : .unordered
    configuration.featureSensitivity = options.highSensitivity ? .high : .normal
    configuration.isObjectMaskingEnabled = options.objectMaskingEnabled

    let session: PhotogrammetrySession
    if let maskDirectory = options.maskDirectory {
        print("Loading \(images.count) keyframes and explicit masks...")
        let samples = try await makeMaskedSamples(images: images, maskDirectory: maskDirectory)
        session = try PhotogrammetrySession(input: samples, configuration: configuration)
    } else {
        print("Loading \(images.count) keyframes; RealityKit will \(options.objectMaskingEnabled ? "attempt automatic foreground masks" : "use complete images")...")
        session = try PhotogrammetrySession(input: options.inputDirectory, configuration: configuration)
    }

    let request = PhotogrammetrySession.Request.modelFile(
        url: options.outputURL,
        detail: options.detail
    )
    try session.process(requests: [request])

    var requestFailed = false
    var completedURL: URL?
    var lastProgress = -1

    for try await output in session.outputs {
        switch output {
        case .inputComplete:
            print("Input ingestion complete.")
        case .requestProgress(_, let fractionComplete):
            let percent = Int((fractionComplete * 100).rounded(.down))
            if percent >= lastProgress + 5 {
                lastProgress = percent
                print("Progress: \(percent)%")
            }
        case .requestProgressInfo(_, let info):
            print("Estimated remaining time: \(formatDuration(info.estimatedRemainingTime))")
        case .requestComplete(_, let result):
            if case .modelFile(let url) = result {
                completedURL = url
                print("Model created: \(url.path)")
            }
        case .requestError(_, let error):
            requestFailed = true
            fputs("Reconstruction request failed: \(error.localizedDescription)\n", stderr)
        case .invalidSample(let id, let reason):
            fputs("Invalid sample \(id): \(reason)\n", stderr)
        case .skippedSample(let id):
            fputs("Skipped sample \(id).\n", stderr)
        case .automaticDownsampling:
            print("RealityKit downsampled inputs to fit available resources.")
        case .stitchingIncomplete:
            fputs("Warning: RealityKit could not stitch every sample into one reconstruction.\n", stderr)
        case .processingCancelled:
            throw CLIError.reconstruction("Object Capture was cancelled.")
        case .processingComplete:
            if requestFailed || completedURL == nil {
                throw CLIError.reconstruction("Object Capture finished without producing a model.")
            }
            return
        @unknown default:
            print("Received an unrecognized Object Capture status update.")
        }
    }

    if requestFailed || completedURL == nil {
        throw CLIError.reconstruction("Object Capture's status stream ended without producing a model.")
    }
}

@main
private struct ObjectCaptureCLI {
    static func main() async {
        do {
            let options = try Options.parse(Array(CommandLine.arguments.dropFirst()))
            try await run(options)
        } catch {
            fputs("error: \(error.localizedDescription)\n", stderr)
            Darwin.exit(EXIT_FAILURE)
        }
    }
}
