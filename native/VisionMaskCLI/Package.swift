// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "VisionMaskCLI",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "vision-mask", targets: ["VisionMaskCLI"])
    ],
    targets: [
        .executableTarget(
            name: "VisionMaskCLI",
            path: "Sources"
        ),
        .testTarget(
            name: "VisionMaskCLITests",
            dependencies: ["VisionMaskCLI"],
            path: "Tests"
        )
    ],
    swiftLanguageModes: [.v5]
)
