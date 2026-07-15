import Foundation

enum CLIError: LocalizedError {
    case help
    case usage(String)
    case invalidInput(String)
    case processing(String)
    case validation(String)

    var errorDescription: String? {
        switch self {
        case .help:
            return nil
        case .usage(let message), .invalidInput(let message), .processing(let message), .validation(let message):
            return message
        }
    }
}

struct Options {
    let inputDirectory: URL
    let outputDirectory: URL
    let debugDirectory: URL?
    let includePattern: String
    let overwrite: Bool
    let dryRun: Bool
    let validateOnly: Bool
    let personThreshold: Float
    let occluderMargin: Int
    let edgeErosion: Int
    let objectSeedX: Double?
    let objectSeedY: Double?
    let occluderExclusionEnabled: Bool

    static let usage = """
    Usage:
      vision-mask --input <keyframe-dir> --output <mask-dir> [options]

    Generates filename-matched, binary 8-bit grayscale PNG object masks with
    Apple's built-in Vision framework. Processing is local and sequential.

    Modes:
      --dry-run                  Decode and preflight inputs; do not run Vision or write files.
      --validate-only           Validate existing masks; do not run Vision or write files.

    Selection and safety:
      --include <glob>           Process matching filenames (default: *).
                                 Quote patterns, for example 'frame_*.jpg'.
      --debug-dir <directory>    Also write review-only RGBA cutouts and mask overlays.
      --person-threshold <0...1> Remove person-mask pixels at or above this confidence
                                 (default: 0.82).
      --occluder-margin <pixels> Dilate person/skin exclusions inward (default: 6).
      --edge-erosion <pixels>    Erode the foreground boundary (default: 1).
      --object-seed <x,y>        Normalized object point (0...1, top-left origin).
                                 Enables category-agnostic seeded selection instead
                                 of the legacy green-tin selector.
      --no-occluder-exclusion    Keep the selected Vision instance intact. Use only
                                 when object color is skin-like and masks are reviewed.
      --overwrite               Replace existing masks and mask_report.json.
      -h, --help                Show this help.

    The selected foreground instance must contain credible green pixels and be
    near the image center. High-confidence person pixels and conservative skin
    pixels are removed before the mask is saved. Ambiguous frames fail closed.

    Example:
      vision-mask --input <keyframes> --output <masks> --include 'frame_*.jpg' --dry-run
    """

    static func parse(_ arguments: [String]) throws -> Options {
        var input: String?
        var output: String?
        var debugDirectory: String?
        var includePattern = "*"
        var overwrite = false
        var dryRun = false
        var validateOnly = false
        var personThreshold: Float = 0.82
        var occluderMargin = 6
        var edgeErosion = 1
        var objectSeedX: Double?
        var objectSeedY: Double?
        var occluderExclusionEnabled = true

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
            case "--output":
                output = try nextValue()
            case "--include":
                includePattern = try nextValue()
            case "--debug-dir":
                debugDirectory = try nextValue()
            case "--person-threshold":
                let value = try nextValue()
                guard let parsed = Float(value), parsed >= 0, parsed <= 1 else {
                    throw CLIError.usage("--person-threshold must be between 0 and 1.")
                }
                personThreshold = parsed
            case "--occluder-margin":
                let value = try nextValue()
                guard let parsed = Int(value), parsed >= 0, parsed <= 100 else {
                    throw CLIError.usage("--occluder-margin must be an integer from 0 through 100.")
                }
                occluderMargin = parsed
            case "--edge-erosion":
                let value = try nextValue()
                guard let parsed = Int(value), parsed >= 0, parsed <= 20 else {
                    throw CLIError.usage("--edge-erosion must be an integer from 0 through 20.")
                }
                edgeErosion = parsed
            case "--object-seed":
                let pieces = try nextValue().split(separator: ",", omittingEmptySubsequences: false)
                guard pieces.count == 2,
                      let x = Double(pieces[0]), let y = Double(pieces[1]),
                      x >= 0, x <= 1, y >= 0, y <= 1 else {
                    throw CLIError.usage("--object-seed must be normalized x,y coordinates from 0 through 1.")
                }
                objectSeedX = x
                objectSeedY = y
            case "--no-occluder-exclusion":
                occluderExclusionEnabled = false
            case "--overwrite":
                overwrite = true
            case "--dry-run":
                dryRun = true
            case "--validate-only":
                validateOnly = true
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
        guard !(dryRun && validateOnly) else {
            throw CLIError.usage("--dry-run and --validate-only are mutually exclusive.")
        }
        guard !validateOnly || !overwrite else {
            throw CLIError.usage("--overwrite has no effect with --validate-only; remove it.")
        }

        let outputURL = URL(fileURLWithPath: output, isDirectory: true).standardizedFileURL
        let debugURL = debugDirectory.map {
            URL(fileURLWithPath: $0, isDirectory: true).standardizedFileURL
        }
        guard debugURL == nil || debugURL != outputURL else {
            throw CLIError.usage("--debug-dir must differ from --output so review PNGs cannot be mistaken for masks.")
        }
        guard !validateOnly || debugURL == nil else {
            throw CLIError.usage("--debug-dir has no effect with --validate-only; remove it.")
        }

        return Options(
            inputDirectory: URL(fileURLWithPath: input, isDirectory: true).standardizedFileURL,
            outputDirectory: outputURL,
            debugDirectory: debugURL,
            includePattern: includePattern,
            overwrite: overwrite,
            dryRun: dryRun,
            validateOnly: validateOnly,
            personThreshold: personThreshold,
            occluderMargin: occluderMargin,
            edgeErosion: edgeErosion,
            objectSeedX: objectSeedX,
            objectSeedY: objectSeedY,
            occluderExclusionEnabled: occluderExclusionEnabled
        )
    }
}
