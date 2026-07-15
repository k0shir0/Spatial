import CoreVideo
import Foundation
import Vision

struct ScalarPlane {
    let width: Int
    let height: Int
    let values: [Float]

    func sample(normalizedX x: Double, normalizedY y: Double) -> Float {
        let pixelX = min(width - 1, max(0, Int(x * Double(width))))
        let pixelY = min(height - 1, max(0, Int(y * Double(height))))
        return values[pixelY * width + pixelX]
    }
}

private func pixelFormatName(_ format: OSType) -> String {
    let bytes: [UInt8] = [
        UInt8((format >> 24) & 0xff),
        UInt8((format >> 16) & 0xff),
        UInt8((format >> 8) & 0xff),
        UInt8(format & 0xff)
    ]
    let printable = bytes.map { $0 >= 32 && $0 <= 126 ? Character(UnicodeScalar($0)) : "?" }
    return "\(String(printable)) (\(format))"
}

func scalarPlane(from pixelBuffer: CVPixelBuffer) throws -> ScalarPlane {
    guard !CVPixelBufferIsPlanar(pixelBuffer) else {
        throw CLIError.processing("Vision returned an unsupported planar mask buffer.")
    }
    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
    var values = [Float](repeating: 0, count: width * height)

    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
        throw CLIError.processing("Vision returned a mask with no readable base address.")
    }

    switch format {
    case kCVPixelFormatType_OneComponent8:
        for y in 0..<height {
            let row = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: UInt8.self)
            for x in 0..<width { values[y * width + x] = Float(row[x]) / 255.0 }
        }
    case kCVPixelFormatType_OneComponent16Half:
        for y in 0..<height {
            let row = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: UInt16.self)
            for x in 0..<width { values[y * width + x] = Float(Float16(bitPattern: row[x])) }
        }
    case kCVPixelFormatType_OneComponent32Float:
        for y in 0..<height {
            let row = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: Float.self)
            for x in 0..<width { values[y * width + x] = row[x] }
        }
    default:
        throw CLIError.processing(
            "Vision returned unsupported scalar mask format \(pixelFormatName(format))."
        )
    }
    return ScalarPlane(width: width, height: height, values: values)
}

private struct LabelPlane {
    let width: Int
    let height: Int
    let labels: [UInt8]

    init(pixelBuffer: CVPixelBuffer) throws {
        guard !CVPixelBufferIsPlanar(pixelBuffer) else {
            throw CLIError.processing("Vision returned an unsupported planar instance-label buffer.")
        }
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        guard format == kCVPixelFormatType_OneComponent8 else {
            throw CLIError.processing(
                "Vision returned unsupported instance-label format \(pixelFormatName(format)); expected 8-bit labels."
            )
        }

        let localWidth = CVPixelBufferGetWidth(pixelBuffer)
        let localHeight = CVPixelBufferGetHeight(pixelBuffer)
        width = localWidth
        height = localHeight
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        var copied = [UInt8](repeating: 0, count: localWidth * localHeight)
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw CLIError.processing("Vision returned instance labels with no readable base address.")
        }
        for y in 0..<localHeight {
            let row = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: UInt8.self)
            copied.withUnsafeMutableBufferPointer { destination in
                destination.baseAddress!.advanced(by: y * localWidth).update(from: row, count: localWidth)
            }
        }
        labels = copied
    }
}

private struct InstanceAccumulator {
    var pixelCount = 0
    var greenCount = 0
    var skinCount = 0
    var personCount = 0
    var sumX = 0.0
    var sumY = 0.0
}

struct InstanceSelection {
    let identifier: Int
    let score: Double
    let areaFraction: Double
    let greenFraction: Double
    let greenShare: Double
    let centerDistance: Double
    let occluderFraction: Double
    let instanceCount: Int
}

private func chooseTinInstance(
    labels: LabelPlane,
    image: LoadedImage,
    personMask: ScalarPlane,
    personThreshold: Float
) throws -> InstanceSelection {
    var accumulators: [Int: InstanceAccumulator] = [:]
    var totalGreen = 0

    for y in 0..<labels.height {
        let normalizedY = (Double(y) + 0.5) / Double(labels.height)
        let sourceY = min(image.metadata.height - 1, Int(normalizedY * Double(image.metadata.height)))
        for x in 0..<labels.width {
            let identifier = Int(labels.labels[y * labels.width + x])
            guard identifier != 0 else { continue }
            let normalizedX = (Double(x) + 0.5) / Double(labels.width)
            let sourceX = min(image.metadata.width - 1, Int(normalizedX * Double(image.metadata.width)))
            let offset = (sourceY * image.metadata.width + sourceX) * 4
            let red = image.rgba[offset]
            let green = image.rgba[offset + 1]
            let blue = image.rgba[offset + 2]
            let likelyGreen = isLikelyGreen(r: red, g: green, b: blue)
            let likelySkin = isLikelySkin(r: red, g: green, b: blue)
            let likelyPerson = personMask.sample(normalizedX: normalizedX, normalizedY: normalizedY) >= personThreshold

            var accumulator = accumulators[identifier, default: InstanceAccumulator()]
            accumulator.pixelCount += 1
            accumulator.sumX += normalizedX
            accumulator.sumY += normalizedY
            if likelyGreen { accumulator.greenCount += 1; totalGreen += 1 }
            if likelySkin { accumulator.skinCount += 1 }
            if likelyPerson { accumulator.personCount += 1 }
            accumulators[identifier] = accumulator
        }
    }

    let analysisPixels = Double(labels.width * labels.height)
    let minimumGreenPixels = max(6, Int(analysisPixels * 0.00004))
    var selections: [InstanceSelection] = []
    selections.reserveCapacity(accumulators.count)

    for (identifier, value) in accumulators where value.pixelCount > 0 {
        let count = Double(value.pixelCount)
        let areaFraction = count / analysisPixels
        let greenFraction = Double(value.greenCount) / count
        let greenShare = totalGreen > 0 ? Double(value.greenCount) / Double(totalGreen) : 0
        let centerX = value.sumX / count
        let centerY = value.sumY / count
        let centerDistance = hypot(centerX - 0.5, centerY - 0.5)
        let centrality = max(0, 1 - centerDistance / 0.55)
        let occluderCount = max(value.skinCount, value.personCount)
        let occluderFraction = Double(occluderCount) / count
        let usefulArea = min(1, areaFraction / 0.08)
        let score = 4.0 * greenShare
            + 2.5 * greenFraction
            + 1.25 * centrality
            + 0.25 * usefulArea
            - 1.5 * occluderFraction

        guard areaFraction >= 0.001, areaFraction <= 0.75 else { continue }
        guard value.greenCount >= minimumGreenPixels, greenFraction >= 0.018 else { continue }
        guard centerDistance <= 0.5 else { continue }
        selections.append(InstanceSelection(
            identifier: identifier,
            score: score,
            areaFraction: areaFraction,
            greenFraction: greenFraction,
            greenShare: greenShare,
            centerDistance: centerDistance,
            occluderFraction: occluderFraction,
            instanceCount: accumulators.count
        ))
    }

    let ranked = selections.sorted { $0.score > $1.score }
    guard let selected = ranked.first, selected.greenShare >= 0.30 else {
        throw CLIError.processing(
            "No centered foreground instance contained enough green evidence. Refusing to emit a likely hand/person mask."
        )
    }
    if let runnerUp = ranked.dropFirst().first,
       runnerUp.greenShare >= 0.15,
       selected.score - runnerUp.score < 0.35 {
        throw CLIError.processing(
            String(
                format: "Foreground selection is ambiguous (instances %d and %d differ by only %.3f). Refusing to guess.",
                selected.identifier,
                runnerUp.identifier,
                selected.score - runnerUp.score
            )
        )
    }
    return selected
}

private func chooseSeededInstance(
    labels: LabelPlane, image: LoadedImage, personMask: ScalarPlane,
    personThreshold: Float, seedX: Double, seedY: Double
) throws -> InstanceSelection {
    let centerX = min(labels.width - 1, max(0, Int(seedX * Double(labels.width))))
    let centerY = min(labels.height - 1, max(0, Int(seedY * Double(labels.height))))
    var nearby: [Int: Int] = [:]
    for y in max(0, centerY - 2)...min(labels.height - 1, centerY + 2) {
        for x in max(0, centerX - 2)...min(labels.width - 1, centerX + 2) {
            let value = Int(labels.labels[y * labels.width + x])
            if value != 0 { nearby[value, default: 0] += 1 }
        }
    }
    guard let identifier = nearby.max(by: { $0.value < $1.value })?.key else {
        throw CLIError.processing("The object seed falls on background in the Vision instance mask.")
    }
    var count = 0
    var greenCount = 0
    var occluderCount = 0
    var instanceIDs = Set<Int>()
    for y in 0..<labels.height {
        let ny = (Double(y) + 0.5) / Double(labels.height)
        let sy = min(image.metadata.height - 1, Int(ny * Double(image.metadata.height)))
        for x in 0..<labels.width {
            let current = Int(labels.labels[y * labels.width + x])
            if current != 0 { instanceIDs.insert(current) }
            guard current == identifier else { continue }
            count += 1
            let nx = (Double(x) + 0.5) / Double(labels.width)
            let sx = min(image.metadata.width - 1, Int(nx * Double(image.metadata.width)))
            let offset = (sy * image.metadata.width + sx) * 4
            let r = image.rgba[offset], g = image.rgba[offset + 1], b = image.rgba[offset + 2]
            if isLikelyGreen(r: r, g: g, b: b) { greenCount += 1 }
            if isLikelySkin(r: r, g: g, b: b)
                || personMask.sample(normalizedX: nx, normalizedY: ny) >= personThreshold {
                occluderCount += 1
            }
        }
    }
    let area = Double(count) / Double(labels.width * labels.height)
    guard count > 0, area >= 0.001, area <= 0.75 else {
        throw CLIError.processing(String(format: "Seeded foreground instance coverage %.3f%% is unsafe.", area * 100))
    }
    return InstanceSelection(
        identifier: identifier, score: 1.0, areaFraction: area,
        greenFraction: Double(greenCount) / Double(count), greenShare: 0,
        centerDistance: hypot(seedX - 0.5, seedY - 0.5),
        occluderFraction: Double(occluderCount) / Double(count),
        instanceCount: instanceIDs.count
    )
}

struct GeneratedMask {
    let pixels: [UInt8]
    let width: Int
    let height: Int
    let selection: InstanceSelection
    let foregroundFraction: Double
    let greenFraction: Double
    let removedFraction: Double
}

func generateMask(for image: LoadedImage, options: Options) throws -> GeneratedMask {
    let foregroundRequest = VNGenerateForegroundInstanceMaskRequest()
    let personRequest = VNGeneratePersonSegmentationRequest()
    personRequest.qualityLevel = .balanced
    personRequest.outputPixelFormat = kCVPixelFormatType_OneComponent8

    let handler = VNImageRequestHandler(cgImage: image.cgImage, orientation: .up, options: [:])
    do {
        // Keep even the two native requests serialized. This deliberately
        // favors bounded peak memory over throughput.
        try handler.perform([foregroundRequest])
        try handler.perform([personRequest])
    } catch {
        throw CLIError.processing("Vision failed for \(image.metadata.url.lastPathComponent): \(error.localizedDescription)")
    }

    guard let foregroundObservation = foregroundRequest.results?.first else {
        throw CLIError.processing("Vision found no separable foreground in \(image.metadata.url.lastPathComponent).")
    }
    guard let personObservation = personRequest.results?.first else {
        throw CLIError.processing("Vision did not return a person-exclusion mask for \(image.metadata.url.lastPathComponent).")
    }

    let labels = try LabelPlane(pixelBuffer: foregroundObservation.instanceMask)
    let personPlane = try scalarPlane(from: personObservation.pixelBuffer)
    let selection: InstanceSelection
    if let seedX = options.objectSeedX, let seedY = options.objectSeedY {
        selection = try chooseSeededInstance(
            labels: labels, image: image, personMask: personPlane,
            personThreshold: options.personThreshold, seedX: seedX, seedY: seedY
        )
    } else {
        selection = try chooseTinInstance(
            labels: labels, image: image, personMask: personPlane,
            personThreshold: options.personThreshold
        )
    }

    let selectedBuffer: CVPixelBuffer
    do {
        selectedBuffer = try foregroundObservation.generateScaledMaskForImage(
            forInstances: IndexSet(integer: selection.identifier),
            from: handler
        )
    } catch {
        throw CLIError.processing("Vision could not scale the selected mask for \(image.metadata.url.lastPathComponent): \(error.localizedDescription)")
    }
    let selectedPlane = try scalarPlane(from: selectedBuffer)
    let width = image.metadata.width
    let height = image.metadata.height

    var base = [UInt8](repeating: 0, count: width * height)
    for y in 0..<height {
        let normalizedY = (Double(y) + 0.5) / Double(height)
        for x in 0..<width {
            let normalizedX = (Double(x) + 0.5) / Double(width)
            if selectedPlane.sample(normalizedX: normalizedX, normalizedY: normalizedY) >= 0.5 {
                base[y * width + x] = 1
            }
        }
    }

    let maximumHolePixels = max(64, Int(Double(width * height) * 0.025))
    base = fillSmallHoles(base, width: width, height: height, maximumHolePixels: maximumHolePixels)
    base = eroded(base, width: width, height: height, radius: options.edgeErosion)
    let baseCount = base.reduce(0) { $0 + Int($1) }
    guard baseCount > 0 else {
        throw CLIError.processing("The selected mask became empty after boundary erosion.")
    }

    var occluders = [UInt8](repeating: 0, count: width * height)
    for y in 0..<height {
        let normalizedY = (Double(y) + 0.5) / Double(height)
        for x in 0..<width {
            let index = y * width + x
            guard base[index] != 0 else { continue }
            let normalizedX = (Double(x) + 0.5) / Double(width)
            let rgbaOffset = index * 4
            let skin = isLikelySkin(
                r: image.rgba[rgbaOffset],
                g: image.rgba[rgbaOffset + 1],
                b: image.rgba[rgbaOffset + 2]
            )
            let person = personPlane.sample(normalizedX: normalizedX, normalizedY: normalizedY) > options.personThreshold
            if options.occluderExclusionEnabled && (skin || person) { occluders[index] = 1 }
        }
    }
    occluders = dilated(occluders, width: width, height: height, radius: options.occluderMargin)

    var refined = [UInt8](repeating: 0, count: base.count)
    for index in refined.indices where base[index] != 0 && occluders[index] == 0 {
        refined[index] = 1
    }
    if let seedX = options.objectSeedX, let seedY = options.objectSeedY {
        let x = min(width - 1, max(0, Int(seedX * Double(width))))
        let y = min(height - 1, max(0, Int(seedY * Double(height))))
        guard refined[y * width + x] != 0 else {
            throw CLIError.processing(
                "The seeded object pixel was removed as an occluder; refusing to substitute a nearby hand/person component."
            )
        }
        refined = retainComponentNearestSeed(
            refined, width: width, height: height, normalizedX: seedX, normalizedY: seedY
        )
    } else {
        refined = retainBestGreenComponent(refined, rgba: image.rgba, width: width, height: height)
    }

    var foregroundCount = 0
    var greenCount = 0
    var output = [UInt8](repeating: 0, count: refined.count)
    for index in refined.indices where refined[index] != 0 {
        foregroundCount += 1
        let rgbaOffset = index * 4
        if isLikelyGreen(
            r: image.rgba[rgbaOffset],
            g: image.rgba[rgbaOffset + 1],
            b: image.rgba[rgbaOffset + 2]
        ) {
            greenCount += 1
        }
        output[index] = 255
    }

    let pixelCount = Double(width * height)
    let foregroundFraction = Double(foregroundCount) / pixelCount
    let greenFraction = foregroundCount > 0 ? Double(greenCount) / Double(foregroundCount) : 0
    let removedFraction = Double(baseCount - foregroundCount) / Double(baseCount)
    let maximumForeground = options.objectSeedX == nil ? 0.35 : 0.50
    guard foregroundFraction >= 0.0015, foregroundFraction <= maximumForeground else {
        throw CLIError.processing(
            String(format: "Refined mask coverage %.3f%% is outside the safe range; refusing to save it.", foregroundFraction * 100)
        )
    }
    guard options.objectSeedX != nil || greenFraction >= 0.015 else {
        throw CLIError.processing(
            String(format: "Refined mask contains only %.3f%% green evidence; refusing to save it.", greenFraction * 100)
        )
    }

    return GeneratedMask(
        pixels: output,
        width: width,
        height: height,
        selection: selection,
        foregroundFraction: foregroundFraction,
        greenFraction: greenFraction,
        removedFraction: removedFraction
    )
}
