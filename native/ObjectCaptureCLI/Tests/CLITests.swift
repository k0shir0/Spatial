import XCTest
@testable import ObjectCaptureCLI

final class OptionsParseTests: XCTestCase {
    func testParsesRequiredArgumentsWithDefaults() throws {
        let options = try Options.parse(["--input", "frames", "--output", "model.usdz"])
        XCTAssertEqual(options.inputDirectory.lastPathComponent, "frames")
        XCTAssertEqual(options.outputURL.lastPathComponent, "model.usdz")
        XCTAssertNil(options.maskDirectory)
        XCTAssertEqual(options.detail, .reduced)
        XCTAssertTrue(options.sequential)
        XCTAssertTrue(options.highSensitivity)
        XCTAssertTrue(options.objectMaskingEnabled)
        XCTAssertFalse(options.force)
    }

    func testParsesEveryFlag() throws {
        let options = try Options.parse([
            "--input", "frames", "--output", "out/model.usdz", "--masks", "masks",
            "--detail", "FULL", "--unordered", "--normal-sensitivity", "--force"
        ])
        XCTAssertEqual(options.maskDirectory?.lastPathComponent, "masks")
        XCTAssertEqual(options.detail, .full)
        XCTAssertFalse(options.sequential)
        XCTAssertFalse(options.highSensitivity)
        XCTAssertTrue(options.force)
    }

    func testRequiresBothInputAndOutput() {
        XCTAssertThrowsError(try Options.parse(["--input", "frames"]))
        XCTAssertThrowsError(try Options.parse(["--output", "model.usdz"]))
    }

    func testRejectsMasksCombinedWithNoObjectMasking() {
        XCTAssertThrowsError(try Options.parse([
            "--input", "frames", "--output", "model.usdz",
            "--masks", "masks", "--no-object-masking"
        ]))
    }

    func testRejectsUnknownArgumentsAndDetailLevels() {
        XCTAssertThrowsError(try Options.parse(["--input", "a", "--output", "b", "--wat"]))
        XCTAssertThrowsError(try Options.parse(["--input", "a", "--output", "b", "--detail", "ultra"]))
    }

    func testRejectsMissingValueAfterAnOption() {
        XCTAssertThrowsError(try Options.parse(["--input"]))
    }
}

final class ImageFileTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("object-capture-cli-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: directory)
    }

    private func touch(_ name: String) throws {
        try Data().write(to: directory.appendingPathComponent(name))
    }

    func testFiltersToSupportedExtensionsAndSortsNaturally() throws {
        try touch("frame_10.jpg")
        try touch("frame_2.JPG")
        try touch("frame_1.png")
        try touch("notes.txt")
        try touch("depth.exr")
        let files = try imageFiles(in: directory)
        XCTAssertEqual(files.map(\.lastPathComponent), ["frame_1.png", "frame_2.JPG", "frame_10.jpg"])
    }

    func testThrowsForAMissingDirectory() {
        let missing = directory.appendingPathComponent("nope", isDirectory: true)
        XCTAssertThrowsError(try imageFiles(in: missing))
    }

    func testMaskIndexKeysByFilenameStem() throws {
        try touch("frame_1.png")
        try touch("frame_2.png")
        let index = try maskIndex(in: directory)
        XCTAssertEqual(Set(index.keys), ["frame_1", "frame_2"])
    }

    func testMaskIndexRejectsDuplicateStems() throws {
        try touch("frame_1.png")
        try touch("frame_1.jpg")
        XCTAssertThrowsError(try maskIndex(in: directory))
    }
}

final class OutputPreparationTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("object-capture-cli-output-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: directory)
    }

    func testThrowsWhenOutputExistsWithoutForce() throws {
        let output = directory.appendingPathComponent("model.usdz")
        try Data("old".utf8).write(to: output)
        XCTAssertThrowsError(try prepareOutputLocation(at: output, force: false))
    }

    func testReplacesAnExistingOutputWithForce() throws {
        let output = directory.appendingPathComponent("model.usdz")
        try Data("old".utf8).write(to: output)
        try prepareOutputLocation(at: output, force: true)
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
    }

    func testCreatesMissingParentDirectories() throws {
        let output = directory.appendingPathComponent("nested/dir/model.usdz")
        try prepareOutputLocation(at: output, force: false)
        var isDirectory: ObjCBool = false
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: output.deletingLastPathComponent().path,
                isDirectory: &isDirectory
            )
        )
        XCTAssertTrue(isDirectory.boolValue)
    }
}

final class FormatDurationTests: XCTestCase {
    func testFormatsSecondsAndMinutes() {
        XCTAssertEqual(formatDuration(42), "42s")
        XCTAssertEqual(formatDuration(125), "2m 5s")
    }

    func testHandlesMissingOrNonFiniteValues() {
        XCTAssertEqual(formatDuration(nil), "unknown")
        XCTAssertEqual(formatDuration(.infinity), "unknown")
    }
}
