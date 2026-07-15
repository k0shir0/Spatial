import XCTest
@testable import VisionMaskCLI

final class MaskMathTests: XCTestCase {
    func testWildcardMatching() {
        XCTAssertTrue(wildcardMatch("frame_*.jpg", "frame_0019.jpg"))
        XCTAssertFalse(wildcardMatch("frame_*.jpg", "contact_sheet.jpg"))
    }

    func testSkinAndGreenClassifiersDoNotOverlapForTargetColors() {
        XCTAssertTrue(isLikelySkin(r: 219, g: 145, b: 129))
        XCTAssertFalse(isLikelyGreen(r: 219, g: 145, b: 129))
        XCTAssertTrue(isLikelyGreen(r: 130, g: 173, b: 49))
        XCTAssertFalse(isLikelySkin(r: 130, g: 173, b: 49))
    }

    func testDilationAndErosion() {
        var point = [UInt8](repeating: 0, count: 25)
        point[12] = 1
        let expanded = dilated(point, width: 5, height: 5, radius: 1)
        XCTAssertEqual(expanded.reduce(0) { $0 + Int($1) }, 9)

        let contracted = eroded(expanded, width: 5, height: 5, radius: 1)
        XCTAssertEqual(contracted.reduce(0) { $0 + Int($1) }, 1)
        XCTAssertEqual(contracted[12], 1)
    }

    func testSmallHoleFillDoesNotFillBorderBackground() {
        let mask: [UInt8] = [
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,
            0, 1, 0, 1, 0,
            0, 1, 1, 1, 0,
            0, 0, 0, 0, 0,
        ]
        let filled = fillSmallHoles(mask, width: 5, height: 5, maximumHolePixels: 1)
        XCTAssertEqual(filled[12], 1)
        XCTAssertEqual(filled[0], 0)
    }

    func testGreenComponentWinsOverLargerNongreenComponent() {
        let width = 6
        let height = 3
        var mask = [UInt8](repeating: 0, count: width * height)
        mask[0] = 1
        mask[1] = 1
        mask[6] = 1
        mask[7] = 1
        mask[8] = 1
        mask[16] = 1
        mask[17] = 1

        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        for index in 0..<(width * height) {
            rgba[index * 4] = 120
            rgba[index * 4 + 1] = 120
            rgba[index * 4 + 2] = 120
            rgba[index * 4 + 3] = 255
        }
        for index in [16, 17] {
            rgba[index * 4] = 100
            rgba[index * 4 + 1] = 170
            rgba[index * 4 + 2] = 40
        }

        let retained = retainBestGreenComponent(mask, rgba: rgba, width: width, height: height)
        XCTAssertEqual(retained.reduce(0) { $0 + Int($1) }, 2)
        XCTAssertEqual(retained[16], 1)
        XCTAssertEqual(retained[17], 1)
    }

    func testMaskStatisticsRequireBinaryValues() {
        let valid = maskStatistics([0, 0, 255, 255])
        XCTAssertTrue(valid.isBinary)
        XCTAssertEqual(valid.foregroundFraction, 0.5)

        XCTAssertFalse(maskStatistics([0, 127, 255]).isBinary)
    }

    func testComponentNearestSeedWinsWithoutColorPrior() {
        let mask: [UInt8] = [
            1, 1, 0, 0, 0, 0,
            1, 1, 0, 0, 1, 1,
            0, 0, 0, 0, 1, 1,
        ]
        let retained = retainComponentNearestSeed(
            mask, width: 6, height: 3, normalizedX: 0.85, normalizedY: 0.5
        )
        XCTAssertEqual(retained.reduce(0) { $0 + Int($1) }, 4)
        XCTAssertEqual(retained[10], 1)
        XCTAssertEqual(retained[0], 0)
    }
}
