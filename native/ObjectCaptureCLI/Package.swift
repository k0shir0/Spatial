// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "ObjectCaptureCLI",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .executable(name: "object-capture", targets: ["ObjectCaptureCLI"])
    ],
    targets: [
        .executableTarget(
            name: "ObjectCaptureCLI",
            path: "Sources"
        ),
        .testTarget(
            name: "ObjectCaptureCLITests",
            dependencies: ["ObjectCaptureCLI"],
            path: "Tests"
        )
    ],
    swiftLanguageModes: [.v5]
)
