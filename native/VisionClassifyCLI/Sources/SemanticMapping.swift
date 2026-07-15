import Foundation

enum SemanticKind: String, CaseIterable, Codable {
    case phone
    case tin
    case bottle
    case can
    case book
}

struct RawLabel: Codable {
    let identifier: String
    let confidence: Float
}

struct MappedEvidence: Codable {
    let kind: SemanticKind
    let identifier: String
    let confidence: Float
}

struct HintAggregate: Codable {
    let kind: SemanticKind
    let supportFrames: Int
    let supportFraction: Double
    let meanSupportingConfidence: Double
    let maximumConfidence: Float
    let score: Double
}

struct SemanticDecision: Codable {
    let status: String
    let hint: SemanticKind?
    let confidence: Double?
    let supportFrames: Int
    let supportFraction: Double
    let runnerUp: SemanticKind?
    let margin: Double?
    let reason: String
}

private func normalizedLabel(_ identifier: String) -> String {
    identifier
        .lowercased()
        .replacingOccurrences(of: "_", with: " ")
        .replacingOccurrences(of: "-", with: " ")
        .split { !$0.isLetter && !$0.isNumber }
        .map(String.init)
        .joined(separator: " ")
}

private func hasPhrase(_ normalized: String, _ phrase: String) -> Bool {
    normalized == phrase
        || normalized.hasPrefix("\(phrase) ")
        || normalized.hasSuffix(" \(phrase)")
        || normalized.contains(" \(phrase) ")
}

/// Maps a deliberately narrow set of classifier labels to advisory object hints.
/// Broad words such as "container", "telephone booth", "watering can", and
/// "can opener" are intentionally not accepted.
func semanticKind(for identifier: String) -> SemanticKind? {
    let label = normalizedLabel(identifier)

    let accessoryWords = [
        "opener", "cap", "cork", "brush", "rack", "holder", "case", "charger",
        "accessory", "booth"
    ]
    if accessoryWords.contains(where: { hasPhrase(label, $0) }) {
        return nil
    }

    let phonePhrases = [
        "cell phone", "cellular phone", "cellular telephone", "mobile phone",
        "mobile telephone", "smartphone", "smart phone", "cellphone"
    ]
    if label == "phone" || label == "telephone"
        || phonePhrases.contains(where: { hasPhrase(label, $0) }) {
        return .phone
    }

    let bottlePhrases = [
        "bottle", "water bottle", "beer bottle", "wine bottle", "soda bottle",
        "plastic bottle", "glass bottle", "medicine bottle"
    ]
    if bottlePhrases.contains(where: { hasPhrase(label, $0) }) {
        return .bottle
    }

    let canPhrases = [
        "beverage can", "soda can", "soft drink can", "aluminum can",
        "aluminium can", "food can", "tin can"
    ]
    if label == "can" || canPhrases.contains(where: { hasPhrase(label, $0) }) {
        return .can
    }

    let tinPhrases = [
        "tin", "mint tin", "metal tin", "tin box", "metal box"
    ]
    if tinPhrases.contains(where: { hasPhrase(label, $0) }) {
        return .tin
    }

    // Keep this list exact. Broad Vision labels such as "document", "paper",
    // "printed page", and "book jacket" are not evidence that the physical
    // object is a book-shaped volume.
    let bookLabels: Set<String> = [
        "book", "book volume", "paperback", "paperback book",
        "hardback", "hardback book", "hardcover", "hardcover book",
        "textbook", "text book", "comic book", "picture book", "reference book"
    ]
    if bookLabels.contains(label) {
        return .book
    }
    return nil
}

func bestMappedEvidence(labels: [RawLabel], minimumConfidence: Float) -> [MappedEvidence] {
    var best: [SemanticKind: RawLabel] = [:]
    for label in labels where label.confidence >= minimumConfidence {
        guard let kind = semanticKind(for: label.identifier) else { continue }
        if best[kind] == nil || label.confidence > best[kind]!.confidence {
            best[kind] = label
        }
    }
    return SemanticKind.allCases.compactMap { kind in
        guard let label = best[kind] else { return nil }
        return MappedEvidence(kind: kind, identifier: label.identifier, confidence: label.confidence)
    }
}

func aggregateHints(
    frameEvidence: [[MappedEvidence]],
    minimumScore: Double,
    minimumSupportFrames: Int,
    minimumSupportFraction: Double,
    minimumWinnerMargin: Double
) -> (aggregates: [HintAggregate], decision: SemanticDecision) {
    let frameCount = frameEvidence.count
    guard frameCount > 0 else {
        return ([], SemanticDecision(
            status: "abstained",
            hint: nil,
            confidence: nil,
            supportFrames: 0,
            supportFraction: 0,
            runnerUp: nil,
            margin: nil,
            reason: "No frames were classified."
        ))
    }

    var aggregates: [HintAggregate] = []
    for kind in SemanticKind.allCases {
        let values = frameEvidence.flatMap { frame in
            frame.filter { $0.kind == kind }.map { $0.confidence }
        }
        guard !values.isEmpty else { continue }
        let supportFrames = values.count
        let supportFraction = Double(supportFrames) / Double(frameCount)
        let mean = values.reduce(0.0) { $0 + Double($1) } / Double(supportFrames)
        // The geometric mean penalizes a label seen in only a tiny fraction of
        // frames while retaining strong evidence from informative viewpoints.
        let score = mean * sqrt(supportFraction)
        aggregates.append(HintAggregate(
            kind: kind,
            supportFrames: supportFrames,
            supportFraction: supportFraction,
            meanSupportingConfidence: mean,
            maximumConfidence: values.max() ?? 0,
            score: score
        ))
    }
    aggregates.sort {
        $0.score == $1.score ? $0.kind.rawValue < $1.kind.rawValue : $0.score > $1.score
    }

    guard let winner = aggregates.first else {
        return (aggregates, SemanticDecision(
            status: "abstained",
            hint: nil,
            confidence: nil,
            supportFrames: 0,
            supportFraction: 0,
            runnerUp: nil,
            margin: nil,
            reason: "No supported phone, tin, bottle, can, or book label exceeded the per-frame evidence threshold."
        ))
    }
    let runnerUp = aggregates.dropFirst().first
    let margin = winner.score - (runnerUp?.score ?? 0)

    var failed: [String] = []
    if winner.score < minimumScore {
        failed.append(String(format: "score %.3f is below %.3f", winner.score, minimumScore))
    }
    if winner.supportFrames < minimumSupportFrames {
        failed.append("support \(winner.supportFrames) frames is below \(minimumSupportFrames)")
    }
    if winner.supportFraction < minimumSupportFraction {
        failed.append(String(
            format: "support fraction %.3f is below %.3f",
            winner.supportFraction,
            minimumSupportFraction
        ))
    }
    if margin < minimumWinnerMargin {
        failed.append(String(format: "winner margin %.3f is below %.3f", margin, minimumWinnerMargin))
    }

    if !failed.isEmpty {
        return (aggregates, SemanticDecision(
            status: "abstained",
            hint: nil,
            confidence: nil,
            supportFrames: winner.supportFrames,
            supportFraction: winner.supportFraction,
            runnerUp: runnerUp?.kind,
            margin: margin,
            reason: "Candidate \(winner.kind.rawValue) rejected: \(failed.joined(separator: "; "))."
        ))
    }
    return (aggregates, SemanticDecision(
        status: "accepted",
        hint: winner.kind,
        confidence: winner.score,
        supportFrames: winner.supportFrames,
        supportFraction: winner.supportFraction,
        runnerUp: runnerUp?.kind,
        margin: margin,
        reason: "High-confidence advisory semantic hint; geometry must still be classified independently."
    ))
}
