import Darwin
import Foundation

enum CLIError: LocalizedError {
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

/// Reconstruction detail requested on the command line. Kept independent of
/// RealityKit so argument parsing stays testable without the framework; the
/// mapping to `PhotogrammetrySession.Request.Detail` lives next to the session.
enum Detail: String {
    case preview
    case reduced
    case medium
    case full
    case raw
}

struct Options {
    let inputDirectory: URL
    let maskDirectory: URL?
    let outputURL: URL
    let detail: Detail
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

        guard let detail = Detail(rawValue: detailName) else {
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

let supportedImageExtensions: Set<String> = [
    "heic", "heif", "jpeg", "jpg", "png", "tif", "tiff"
]

func imageFiles(in directory: URL) throws -> [URL] {
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

func maskIndex(in directory: URL) throws -> [String: URL] {
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

func formatDuration(_ seconds: TimeInterval?) -> String {
    guard let seconds, seconds.isFinite else { return "unknown" }
    if seconds < 60 { return "\(Int(seconds.rounded()))s" }
    return "\(Int(seconds / 60))m \(Int(seconds.truncatingRemainder(dividingBy: 60).rounded()))s"
}

func prepareOutputLocation(at outputURL: URL, force: Bool) throws {
    let fileManager = FileManager.default
    if fileManager.fileExists(atPath: outputURL.path) {
        guard force else {
            throw CLIError.invalidInput("Output already exists: \(outputURL.path). Pass --force to replace it.")
        }
        try fileManager.removeItem(at: outputURL)
    }
    try fileManager.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
}
