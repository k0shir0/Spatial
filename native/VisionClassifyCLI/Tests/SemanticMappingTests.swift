import XCTest
@testable import VisionClassifyCLI

final class SemanticMappingTests: XCTestCase {
    func testWildcardMatching() {
        XCTAssertTrue(wildcardMatch("frame_*.jpg", "frame_0019.jpg"))
        XCTAssertFalse(wildcardMatch("frame_*.jpg", "contact_sheet.jpg"))
    }

    func testConservativeSemanticMapping() {
        XCTAssertEqual(semanticKind(for: "cellular telephone"), .phone)
        XCTAssertEqual(semanticKind(for: "smartphone"), .phone)
        XCTAssertEqual(semanticKind(for: "phone"), .phone)
        XCTAssertEqual(semanticKind(for: "water bottle"), .bottle)
        XCTAssertEqual(semanticKind(for: "tin"), .tin)
        XCTAssertEqual(semanticKind(for: "soda can"), .can)
        XCTAssertEqual(semanticKind(for: "can"), .can)
        XCTAssertEqual(semanticKind(for: "book"), .book)
        XCTAssertEqual(semanticKind(for: "paperback"), .book)
        XCTAssertEqual(semanticKind(for: "textbook"), .book)
        XCTAssertEqual(semanticKind(for: "hardcover book"), .book)
        XCTAssertNil(semanticKind(for: "watering can"))
        XCTAssertNil(semanticKind(for: "can opener"))
        XCTAssertNil(semanticKind(for: "bottle opener"))
        XCTAssertNil(semanticKind(for: "phone case"))
        XCTAssertNil(semanticKind(for: "telephone booth"))
        XCTAssertNil(semanticKind(for: "container"))
        XCTAssertNil(semanticKind(for: "document"))
        XCTAssertNil(semanticKind(for: "printed page"))
        XCTAssertNil(semanticKind(for: "paper"))
        XCTAssertNil(semanticKind(for: "book jacket"))
        XCTAssertNil(semanticKind(for: "bookcase"))
        XCTAssertNil(semanticKind(for: "notebook computer"))
    }

    func testEvidenceKeepsOnlyBestLabelPerKind() {
        let evidence = bestMappedEvidence(
            labels: [
                RawLabel(identifier: "smart phone", confidence: 0.35),
                RawLabel(identifier: "cellular telephone", confidence: 0.72),
                RawLabel(identifier: "water bottle", confidence: 0.15)
            ],
            minimumConfidence: 0.20
        )
        XCTAssertEqual(evidence.count, 1)
        XCTAssertEqual(evidence.first?.kind, .phone)
        XCTAssertEqual(evidence.first?.confidence, 0.72)
    }

    func testAggregateAcceptsRepeatedStrongHint() {
        let phone = MappedEvidence(kind: .phone, identifier: "smartphone", confidence: 0.80)
        let empty: [MappedEvidence] = []
        let result = aggregateHints(
            frameEvidence: [[phone], [phone], [phone], empty, empty, empty],
            minimumScore: 0.30,
            minimumSupportFrames: 2,
            minimumSupportFraction: 0.15,
            minimumWinnerMargin: 0.08
        )
        XCTAssertEqual(result.decision.status, "accepted")
        XCTAssertEqual(result.decision.hint, .phone)
    }

    func testAggregateAppliesExistingGatesToBookHint() {
        let book = MappedEvidence(kind: .book, identifier: "paperback", confidence: 0.82)
        let empty: [MappedEvidence] = []
        let accepted = aggregateHints(
            frameEvidence: [[book], [book], [book], empty, empty, empty],
            minimumScore: 0.30,
            minimumSupportFrames: 2,
            minimumSupportFraction: 0.15,
            minimumWinnerMargin: 0.08
        )
        XCTAssertEqual(accepted.decision.status, "accepted")
        XCTAssertEqual(accepted.decision.hint, .book)

        let sparse = aggregateHints(
            frameEvidence: [[book], empty, empty, empty, empty, empty],
            minimumScore: 0.20,
            minimumSupportFrames: 2,
            minimumSupportFraction: 0.15,
            minimumWinnerMargin: 0.08
        )
        XCTAssertEqual(sparse.decision.status, "abstained")
        XCTAssertNil(sparse.decision.hint)
    }

    func testAggregateAbstainsOnSingleFrameAndAmbiguousWinner() {
        let phone = MappedEvidence(kind: .phone, identifier: "smartphone", confidence: 0.90)
        let sparse = aggregateHints(
            frameEvidence: [[phone], [], [], [], [], [], [], []],
            minimumScore: 0.20,
            minimumSupportFrames: 2,
            minimumSupportFraction: 0.15,
            minimumWinnerMargin: 0.08
        )
        XCTAssertEqual(sparse.decision.status, "abstained")

        let tin = MappedEvidence(kind: .tin, identifier: "tin", confidence: 0.85)
        let ambiguous = aggregateHints(
            frameEvidence: [[phone, tin], [phone, tin], [], []],
            minimumScore: 0.20,
            minimumSupportFrames: 2,
            minimumSupportFraction: 0.15,
            minimumWinnerMargin: 0.08
        )
        XCTAssertEqual(ambiguous.decision.status, "abstained")
    }
}
