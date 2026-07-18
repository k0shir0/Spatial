import CoreGraphics
import CoreImage
import CoreVideo
import Darwin
import Foundation
import ImageIO
import RealityKit

private extension PhotogrammetrySession.Request.Detail {
    init(_ detail: Detail) {
        switch detail {
        case .preview: self = .preview
        case .reduced: self = .reduced
        case .medium: self = .medium
        case .full: self = .full
        case .raw: self = .raw
        }
    }
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

private func run(_ options: Options) async throws {
    guard PhotogrammetrySession.isSupported else { throw CLIError.unsupported }

    let images = try imageFiles(in: options.inputDirectory)
    guard images.count >= 3 else {
        throw CLIError.invalidInput("Object Capture needs at least 3 usable keyframes; found \(images.count) in \(options.inputDirectory.path).")
    }

    try prepareOutputLocation(at: options.outputURL, force: options.force)

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
        detail: .init(options.detail)
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
