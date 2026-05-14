import Foundation

public enum SwingSource: String, Codable, Equatable {
    case recorded
    case uploaded

    public var label: String {
        switch self {
        case .recorded:
            "Recorded"
        case .uploaded:
            "Uploaded"
        }
    }
}

public enum SwingOrientation: String, Codable, Equatable {
    case portrait
    case landscape
    case unknown
}

public enum GolfClub: String, Codable, CaseIterable, Identifiable, Equatable {
    case driver = "D"
    case threeWood = "3W"
    case fiveWood = "5W"
    case threeHybrid = "3H"
    case fourHybrid = "4H"
    case fiveHybrid = "5H"
    case threeIron = "3"
    case fourIron = "4"
    case fiveIron = "5"
    case sixIron = "6"
    case sevenIron = "7"
    case eightIron = "8"
    case nineIron = "9"
    case pitchingWedge = "PW"
    case approachWedge = "AW"
    case sandWedge = "SW"
    case lobWedge = "LW"
    case other = "Other"

    public var id: String { rawValue }

    public static let groups: [(title: String, clubs: [GolfClub])] = [
        ("Wood", [.driver, .threeWood, .fiveWood]),
        ("Hybrid", [.threeHybrid, .fourHybrid, .fiveHybrid]),
        ("Iron", [.threeIron, .fourIron, .fiveIron, .sixIron, .sevenIron, .eightIron, .nineIron]),
        ("Wedge", [.pitchingWedge, .approachWedge, .sandWedge, .lobWedge]),
        ("Other", [.other]),
    ]

    public static func canonical(_ value: String?) -> GolfClub? {
        guard let value else { return nil }
        return GolfClub(rawValue: value.trimmingCharacters(in: .whitespacesAndNewlines))
    }
}

public struct VideoEditState: Codable, Equatable {
    public static let identity = VideoEditState()

    public var rotationDegrees: Int
    public var isMirroredHorizontally: Bool

    private enum CodingKeys: String, CodingKey {
        case rotationDegrees
        case isMirroredHorizontally
    }

    public init(rotationDegrees: Int = 0, isMirroredHorizontally: Bool = false) {
        self.rotationDegrees = Self.normalizedRotationDegrees(rotationDegrees)
        self.isMirroredHorizontally = isMirroredHorizontally
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let rotationDegrees = try container.decodeIfPresent(Int.self, forKey: .rotationDegrees) ?? 0
        let isMirroredHorizontally = try container.decodeIfPresent(Bool.self, forKey: .isMirroredHorizontally) ?? false
        self.init(rotationDegrees: rotationDegrees, isMirroredHorizontally: isMirroredHorizontally)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(rotationDegrees, forKey: .rotationDegrees)
        try container.encode(isMirroredHorizontally, forKey: .isMirroredHorizontally)
    }

    public var isIdentity: Bool {
        rotationDegrees == 0 && !isMirroredHorizontally
    }

    public var rotatesDimensions: Bool {
        rotationDegrees == 90 || rotationDegrees == 270
    }

    public var normalized: VideoEditState {
        VideoEditState(rotationDegrees: rotationDegrees, isMirroredHorizontally: isMirroredHorizontally)
    }

    public func rotatedClockwise() -> VideoEditState {
        VideoEditState(
            rotationDegrees: rotationDegrees + 90,
            isMirroredHorizontally: isMirroredHorizontally
        )
    }

    public func togglingHorizontalMirror() -> VideoEditState {
        VideoEditState(
            rotationDegrees: rotationDegrees,
            isMirroredHorizontally: !isMirroredHorizontally
        )
    }

    private static func normalizedRotationDegrees(_ degrees: Int) -> Int {
        let normalized = ((degrees % 360) + 360) % 360
        let quarterTurns = Int((Double(normalized) / 90.0).rounded()) % 4
        return quarterTurns * 90
    }
}

public enum SwingStatus: String, Codable, Equatable {
    case trimming
    case pending
    case uploading
    case analyzing
    case complete
    case failed

    public var label: String {
        switch self {
        case .trimming:
            "Trimming"
        case .pending:
            "Ready"
        case .uploading:
            "Uploading"
        case .analyzing:
            "Analyzing"
        case .complete:
            "Complete"
        case .failed:
            "Failed"
        }
    }
}

public struct SwingRecord: Codable, Equatable, Identifiable {
    public var id: String
    public var videoURL: URL
    public var createdAt: Date
    public var source: SwingSource
    public var orientation: SwingOrientation
    public var durationMs: Int?
    public var trimStartMs: Int?
    public var trimEndMs: Int?
    public var videoEditState: VideoEditState?
    public var club: String?
    public var aiAnalysis: SwingAIFeedback?
    public var senseiScore: Double?
    public var analysis: SwingAnalysis?
    public var analysisError: String?
    public var status: SwingStatus

    public init(
        id: String,
        videoURL: URL,
        createdAt: Date = Date(),
        source: SwingSource,
        orientation: SwingOrientation = .unknown,
        durationMs: Int? = nil,
        trimStartMs: Int? = nil,
        trimEndMs: Int? = nil,
        videoEditState: VideoEditState? = nil,
        club: String? = nil,
        aiAnalysis: SwingAIFeedback? = nil,
        senseiScore: Double? = nil,
        analysis: SwingAnalysis? = nil,
        analysisError: String? = nil,
        status: SwingStatus
    ) {
        self.id = id
        self.videoURL = videoURL
        self.createdAt = createdAt
        self.source = source
        self.orientation = orientation
        self.durationMs = durationMs
        self.trimStartMs = trimStartMs
        self.trimEndMs = trimEndMs
        self.videoEditState = videoEditState
        self.club = club
        self.aiAnalysis = aiAnalysis
        self.senseiScore = senseiScore
        self.analysis = analysis
        self.analysisError = analysisError
        self.status = status
    }
}

public struct TrimWindow: Codable, Equatable {
    public let startMs: Int
    public let endMs: Int

    public var durationMs: Int {
        max(0, endMs - startMs)
    }

    public init(startMs: Int, endMs: Int) {
        self.startMs = startMs
        self.endMs = endMs
    }

    public static func validated(
        startMs: Double,
        endMs: Double,
        durationMs: Int,
        minimumDurationMs: Int? = nil
    ) -> TrimWindow {
        let safeDuration = max(1, durationMs)
        let minimum = max(1, minimumDurationMs ?? min(800, safeDuration))
        let boundedStart = max(0, min(Int(startMs.rounded()), max(0, safeDuration - minimum)))
        let boundedEnd = max(boundedStart + minimum, min(Int(endMs.rounded()), safeDuration))
        return TrimWindow(startMs: boundedStart, endMs: boundedEnd)
    }

    public static func canContinue(startMs: Double, endMs: Double, durationMs: Int) -> Bool {
        guard durationMs > 0 else { return false }
        let minimum = min(800, durationMs)
        return Int((endMs - startMs).rounded()) >= minimum
    }
}
