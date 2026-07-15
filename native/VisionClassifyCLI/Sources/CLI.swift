import Foundation

enum CLIError: LocalizedError {
    case help
    case usage(String)
    case invalidInput(String)
    case processing(String)

    var errorDescription: String? {
        switch self {
        case .help:
            return nil
        case .usage(let message), .invalidInput(let message), .processing(let message):
            return message
        }
    }
}

struct Options {
    let inputDirectory: URL
    let maskDirectory: URL?
    let outputFile: URL
    let includePattern: String
    let topK: Int
    let minimumLabelConfidence: Float
    let minimumEvidenceConfidence: Float
    let minimumHintScore: Double
    let minimumSupportFrames: Int
    let minimumSupportFraction: Double
    let minimumWinnerMargin: Double
    let maskPaddingFraction: Double
    let overwrite: Bool
    let dryRun: Bool

    static let usage = """
    Usage:
      vision-classify --input <image-dir> --output <report.json> [options]

    Runs Apple's built-in Vision image classifier sequentially and writes
    advisory semantic evidence. It never selects or dispatches geometry.

    Input:
      --masks <mask-dir>             Use same-stem binary PNG object masks. Images
                                     without a mask are skipped and reported.
      --include <glob>               Select image filenames (default: *).
      --mask-padding <0...1>         Padding around the masked object (default: 0.12).

    Reporting and conservative hint gates:
      --top-k <1...50>               Per-frame labels retained in JSON (default: 8).
      --min-label-confidence <0...1> Minimum retained raw label (default: 0.01).
      --min-evidence-confidence <0...1>
                                     Minimum mapped label evidence (default: 0.20).
      --min-hint-score <0...1>       Minimum aggregate hint score (default: 0.30).
      --min-support-frames <1...100> Minimum supporting frames (default: 2).
      --min-support-fraction <0...1> Minimum supporting-frame fraction (default: 0.15).
      --min-winner-margin <0...1>    Minimum lead over the runner-up (default: 0.08).

    Modes:
      --dry-run                      Validate inputs without Vision inference or writes.
      --overwrite                    Replace an existing output JSON.
      -h, --help                     Show this help.

    Only phone, tin, bottle, cylindrical-can, and book hints can be emitted. A
    hint is optional and advisory; a geometry router must independently validate
    shape.
    """

    static func parse(_ arguments: [String]) throws -> Options {
        var input: String?
        var masks: String?
        var output: String?
        var includePattern = "*"
        var topK = 8
        var minimumLabelConfidence: Float = 0.01
        var minimumEvidenceConfidence: Float = 0.20
        var minimumHintScore = 0.30
        var minimumSupportFrames = 2
        var minimumSupportFraction = 0.15
        var minimumWinnerMargin = 0.08
        var maskPaddingFraction = 0.12
        var overwrite = false
        var dryRun = false

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

            func probability(_ name: String) throws -> Double {
                let text = try nextValue()
                guard let value = Double(text), value.isFinite, value >= 0, value <= 1 else {
                    throw CLIError.usage("\(name) must be between 0 and 1.")
                }
                return value
            }

            switch argument {
            case "--input":
                input = try nextValue()
            case "--masks":
                masks = try nextValue()
            case "--output":
                output = try nextValue()
            case "--include":
                includePattern = try nextValue()
            case "--top-k":
                let text = try nextValue()
                guard let value = Int(text), (1...50).contains(value) else {
                    throw CLIError.usage("--top-k must be an integer from 1 through 50.")
                }
                topK = value
            case "--min-label-confidence":
                minimumLabelConfidence = Float(try probability(argument))
            case "--min-evidence-confidence":
                minimumEvidenceConfidence = Float(try probability(argument))
            case "--min-hint-score":
                minimumHintScore = try probability(argument)
            case "--min-support-frames":
                let text = try nextValue()
                guard let value = Int(text), (1...100).contains(value) else {
                    throw CLIError.usage("--min-support-frames must be an integer from 1 through 100.")
                }
                minimumSupportFrames = value
            case "--min-support-fraction":
                minimumSupportFraction = try probability(argument)
            case "--min-winner-margin":
                minimumWinnerMargin = try probability(argument)
            case "--mask-padding":
                maskPaddingFraction = try probability(argument)
            case "--overwrite":
                overwrite = true
            case "--dry-run":
                dryRun = true
            case "-h", "--help":
                throw CLIError.help
            default:
                throw CLIError.usage("Unknown argument: \(argument)\n\n\(usage)")
            }
            index += 1
        }

        guard let input, let output else {
            throw CLIError.usage("Both --input and --output are required.\n\n\(usage)")
        }
        guard !includePattern.isEmpty else {
            throw CLIError.usage("--include cannot be empty.")
        }
        guard minimumEvidenceConfidence >= minimumLabelConfidence else {
            throw CLIError.usage(
                "--min-evidence-confidence must be at least --min-label-confidence."
            )
        }
        guard !(dryRun && overwrite) else {
            throw CLIError.usage("--overwrite has no effect with --dry-run; remove it.")
        }

        return Options(
            inputDirectory: URL(fileURLWithPath: input, isDirectory: true).standardizedFileURL,
            maskDirectory: masks.map {
                URL(fileURLWithPath: $0, isDirectory: true).standardizedFileURL
            },
            outputFile: URL(fileURLWithPath: output).standardizedFileURL,
            includePattern: includePattern,
            topK: topK,
            minimumLabelConfidence: minimumLabelConfidence,
            minimumEvidenceConfidence: minimumEvidenceConfidence,
            minimumHintScore: minimumHintScore,
            minimumSupportFrames: minimumSupportFrames,
            minimumSupportFraction: minimumSupportFraction,
            minimumWinnerMargin: minimumWinnerMargin,
            maskPaddingFraction: maskPaddingFraction,
            overwrite: overwrite,
            dryRun: dryRun
        )
    }
}
