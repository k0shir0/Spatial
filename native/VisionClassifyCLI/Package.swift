// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "VisionClassifyCLI",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "vision-classify", targets: ["VisionClassifyCLI"])
    ],
    targets: [
        .executableTarget(
            name: "VisionClassifyCLI",
            path: "Sources"
        ),
        .testTarget(
            name: "VisionClassifyCLITests",
            dependencies: ["VisionClassifyCLI"],
            path: "Tests"
        )
    ],
    swiftLanguageModes: [.v5]
)
