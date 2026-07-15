import Darwin
import Foundation
import Vision

struct FrameReport: Codable {
    let source: String
    let mask: String?
    let foregroundFraction: Double?
    let cropXYWH: [Int]?
    let labels: [RawLabel]
    let mappedEvidence: [MappedEvidence]
}

struct SkippedImage: Codable {
    let source: String
    let reason: String
}

struct AggregateLabel: Codable {
    let identifier: String
    let supportFrames: Int
    let supportFraction: Double
    let meanAcrossAllFrames: Double
    let maximumConfidence: Float
}

struct ReportInput: Codable {
    let imageDirectory: String
    let maskDirectory: String?
    let includePattern: String
    let discoveredImages: Int
    let classifiedImages: Int
}

struct ReportSettings: Codable {
    let topK: Int
    let minimumLabelConfidence: Float
    let minimumEvidenceConfidence: Float
    let minimumHintScore: Double
    let minimumSupportFrames: Int
    let minimumSupportFraction: Double
    let minimumWinnerMargin: Double
    let maskPaddingFraction: Double
}

struct SafetyContract: Encodable {
    let advisoryOnly = true
    let dispatchesGeometry = false
    let overridesGeometry = false
    let allowedHints = SemanticKind.allCases.map(\.rawValue)
}

struct ExecutionReport: Encodable {
    let localOnly = true
    let networkAccess = false
    let downloadedWeights = false
    let bundledModelWeights = false
    let torch = false
    let learnedInference = true
    let modelProvisioning = "operating-system Apple Vision"
    let classifier = "Apple Vision VNClassifyImageRequest"
}

struct RunReport: Encodable {
    let schemaVersion = "1.0"
    let backend = "apple-vision-semantic-hints-v1"
    let createdUTC: String
    let execution = ExecutionReport()
    let contract = SafetyContract()
    let input: ReportInput
    let settings: ReportSettings
    let frames: [FrameReport]
    let skippedImages: [SkippedImage]
    let aggregatedLabels: [AggregateLabel]
    let hintCandidates: [HintAggregate]
    let semanticHint: SemanticDecision
    let warnings: [String]
}

func classify(_ image: PreparedImage, topK: Int, minimumConfidence: Float) throws -> [RawLabel] {
    let request = VNClassifyImageRequest()
    let handler = VNImageRequestHandler(cgImage: image.cgImage, orientation: .up, options: [:])
    do {
        try handler.perform([request])
    } catch {
        throw CLIError.processing("Vision classification failed: \(error.localizedDescription)")
    }
    guard let observations = request.results, !observations.isEmpty else {
        throw CLIError.processing("Vision returned no classification observations.")
    }
    return observations
        .filter { $0.confidence >= minimumConfidence }
        .prefix(topK)
        .map { RawLabel(identifier: $0.identifier, confidence: $0.confidence) }
}

func aggregateRawLabels(_ frames: [FrameReport]) -> [AggregateLabel] {
    guard !frames.isEmpty else { return [] }
    var values: [String: [Float]] = [:]
    for frame in frames {
        for label in frame.labels {
            values[label.identifier, default: []].append(label.confidence)
        }
    }
    return values.map { identifier, confidences in
        let total = confidences.reduce(0.0) { $0 + Double($1) }
        return AggregateLabel(
            identifier: identifier,
            supportFrames: confidences.count,
            supportFraction: Double(confidences.count) / Double(frames.count),
            meanAcrossAllFrames: total / Double(frames.count),
            maximumConfidence: confidences.max() ?? 0
        )
    }.sorted {
        $0.meanAcrossAllFrames == $1.meanAcrossAllFrames
            ? $0.identifier < $1.identifier
            : $0.meanAcrossAllFrames > $1.meanAcrossAllFrames
    }.prefix(30).map { $0 }
}

func preflight(_ options: Options) throws -> (images: [URL], masks: [String: URL]?) {
    let images = try discoverImages(in: options.inputDirectory, includePattern: options.includePattern)
    let masks = try options.maskDirectory.map(discoverMasks)
    if FileManager.default.fileExists(atPath: options.outputFile.path), !options.overwrite {
        throw CLIError.invalidInput(
            "Output already exists: \(options.outputFile.path). Pass --overwrite to replace it."
        )
    }
    let parent = options.outputFile.deletingLastPathComponent()
    var isDirectory: ObjCBool = false
    if FileManager.default.fileExists(atPath: parent.path, isDirectory: &isDirectory),
       !isDirectory.boolValue {
        throw CLIError.invalidInput("Output parent is not a directory: \(parent.path)")
    }
    return (images, masks)
}

func run(_ options: Options) throws {
    let inputs = try preflight(options)
    if options.dryRun {
        var matched = 0
        for imageURL in inputs.images {
            let stem = imageURL.deletingPathExtension().lastPathComponent.lowercased()
            let maskURL = inputs.masks?[stem]
            if inputs.masks != nil, maskURL == nil { continue }
            try autoreleasepool {
                let image = try loadRGBAImage(imageURL)
                let mask = try maskURL.map {
                    try loadMask($0, expectedWidth: image.width, expectedHeight: image.height)
                }
                _ = try prepareImage(
                    image,
                    mask: mask,
                    paddingFraction: options.maskPaddingFraction
                )
            }
            matched += 1
        }
        print("Dry run: \(inputs.images.count) images discovered; \(matched) decoded and validated.")
        print("No Vision requests ran and no files were written.")
        return
    }

    var frames: [FrameReport] = []
    var skipped: [SkippedImage] = []
    for (offset, imageURL) in inputs.images.enumerated() {
        let stem = imageURL.deletingPathExtension().lastPathComponent.lowercased()
        let maskURL = inputs.masks?[stem]
        if inputs.masks != nil, maskURL == nil {
            skipped.append(SkippedImage(
                source: imageURL.path,
                reason: "No same-stem PNG mask; image was not classified."
            ))
            continue
        }

        let frame: FrameReport = try autoreleasepool {
            let image = try loadRGBAImage(imageURL)
            let mask = try maskURL.map {
                try loadMask($0, expectedWidth: image.width, expectedHeight: image.height)
            }
            let prepared = try prepareImage(
                image,
                mask: mask,
                paddingFraction: options.maskPaddingFraction
            )
            let labels = try classify(
                prepared,
                topK: options.topK,
                minimumConfidence: options.minimumLabelConfidence
            )
            return FrameReport(
                source: imageURL.path,
                mask: prepared.maskPath,
                foregroundFraction: prepared.foregroundFraction,
                cropXYWH: prepared.cropXYWH,
                labels: labels,
                mappedEvidence: bestMappedEvidence(
                    labels: labels,
                    minimumConfidence: options.minimumEvidenceConfidence
                )
            )
        }
        frames.append(frame)
        let best = frame.labels.first
        print(String(
            format: "[%d/%d] %@ | %@ %.1f%%",
            offset + 1,
            inputs.images.count,
            imageURL.lastPathComponent,
            best?.identifier ?? "no retained label",
            Double(best?.confidence ?? 0) * 100
        ))
    }

    guard !frames.isEmpty else {
        throw CLIError.invalidInput("No images remained after optional mask matching.")
    }
    let hintResult = aggregateHints(
        frameEvidence: frames.map(\.mappedEvidence),
        minimumScore: options.minimumHintScore,
        minimumSupportFrames: options.minimumSupportFrames,
        minimumSupportFraction: options.minimumSupportFraction,
        minimumWinnerMargin: options.minimumWinnerMargin
    )
    var warnings: [String] = []
    if !skipped.isEmpty {
        warnings.append("\(skipped.count) images lacked a same-stem mask and were skipped.")
    }
    if options.maskDirectory == nil {
        warnings.append("No masks were supplied; background and hands may dominate semantic labels.")
    }
    warnings.append("Semantic labels are advisory and must never override silhouette geometry.")

    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    let report = RunReport(
        createdUTC: formatter.string(from: Date()),
        input: ReportInput(
            imageDirectory: options.inputDirectory.path,
            maskDirectory: options.maskDirectory?.path,
            includePattern: options.includePattern,
            discoveredImages: inputs.images.count,
            classifiedImages: frames.count
        ),
        settings: ReportSettings(
            topK: options.topK,
            minimumLabelConfidence: options.minimumLabelConfidence,
            minimumEvidenceConfidence: options.minimumEvidenceConfidence,
            minimumHintScore: options.minimumHintScore,
            minimumSupportFrames: options.minimumSupportFrames,
            minimumSupportFraction: options.minimumSupportFraction,
            minimumWinnerMargin: options.minimumWinnerMargin,
            maskPaddingFraction: options.maskPaddingFraction
        ),
        frames: frames,
        skippedImages: skipped,
        aggregatedLabels: aggregateRawLabels(frames),
        hintCandidates: hintResult.aggregates,
        semanticHint: hintResult.decision,
        warnings: warnings
    )

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let data = try encoder.encode(report) + Data([0x0a])
    try FileManager.default.createDirectory(
        at: options.outputFile.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try data.write(to: options.outputFile, options: .atomic)
    print("Report: \(options.outputFile.path)")
    print("Semantic hint: \(hintResult.decision.hint?.rawValue ?? "none") (\(hintResult.decision.status))")
}

do {
    let options = try Options.parse(Array(CommandLine.arguments.dropFirst()))
    try run(options)
} catch CLIError.help {
    print(Options.usage)
    Darwin.exit(EXIT_SUCCESS)
} catch {
    fputs("vision-classify: \(error.localizedDescription)\n", stderr)
    Darwin.exit(EXIT_FAILURE)
}
