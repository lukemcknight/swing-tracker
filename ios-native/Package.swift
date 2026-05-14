// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "SwingSenseiNativeCore",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(name: "SwingSenseiCore", targets: ["SwingSenseiCore"]),
    ],
    targets: [
        .target(
            name: "SwingSenseiCore",
            path: "Sources/SwingSenseiCore"
        ),
        .testTarget(
            name: "SwingSenseiCoreTests",
            dependencies: ["SwingSenseiCore"],
            path: "Tests/SwingSenseiCoreTests"
        ),
    ]
)
