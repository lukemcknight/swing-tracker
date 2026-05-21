import Foundation

public enum SwingPosition: String, CaseIterable, Codable, Identifiable {
    case address
    case takeaway
    case top
    case impact
    case finish

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .address:
            "Address"
        case .takeaway:
            "Takeaway"
        case .top:
            "Top"
        case .impact:
            "Impact"
        case .finish:
            "Finish"
        }
    }
}

public struct SwingPoint: Codable, Equatable {
    public let x: Double
    public let y: Double

    public init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }
}

public struct SwingKeypoint: Codable, Equatable, Identifiable {
    public var id: String { name }

    public let name: String
    public let x: Double
    public let y: Double
    public let score: Double

    public init(name: String, x: Double, y: Double, score: Double) {
        self.name = name
        self.x = x
        self.y = y
        self.score = score
    }
}

public struct SwingAnalysisFrame: Codable, Equatable, Identifiable {
    public var id: Int { t }

    public let t: Int
    public let keypoints: [SwingKeypoint]
    public let clubHead: SwingPoint?

    public init(t: Int, keypoints: [SwingKeypoint], clubHead: SwingPoint? = nil) {
        self.t = t
        self.keypoints = keypoints
        self.clubHead = clubHead
    }
}

public struct SwingMetrics: Codable, Equatable {
    public let headDisplacement: Double
    public let shoulderTurn: Double
    public let hipSway: Double
    public let postureLoss: Double
    public let tempo: Double
}

public struct SpeedRange: Codable, Equatable {
    public let min: Double
    public let max: Double
    public let unit: String

    public init(min: Double, max: Double, unit: String = "mph") {
        self.min = min
        self.max = max
        self.unit = unit
    }
}

public struct ShotEstimateEvidence: Codable, Equatable {
    public let sourceFps: Double
    public let calibrationSource: String
    public let detectedBallFlightFrames: Int

    public init(sourceFps: Double, calibrationSource: String, detectedBallFlightFrames: Int) {
        self.sourceFps = sourceFps
        self.calibrationSource = calibrationSource
        self.detectedBallFlightFrames = detectedBallFlightFrames
    }
}

public struct ShotEstimate: Codable, Equatable {
    public let clubSpeedMphRange: SpeedRange?
    public let ballSpeedMphRange: SpeedRange?
    public let landingZone: String
    public let confidence: String
    public let evidence: ShotEstimateEvidence
    public let limitations: [String]

    public init(
        clubSpeedMphRange: SpeedRange? = nil,
        ballSpeedMphRange: SpeedRange? = nil,
        landingZone: String,
        confidence: String,
        evidence: ShotEstimateEvidence,
        limitations: [String]
    ) {
        self.clubSpeedMphRange = clubSpeedMphRange
        self.ballSpeedMphRange = ballSpeedMphRange
        self.landingZone = landingZone
        self.confidence = confidence
        self.evidence = evidence
        self.limitations = limitations
    }
}

public enum FeedbackStatus: String, Codable {
    case good
    case warning
    case bad
}

public struct FeedbackItem: Codable, Equatable, Identifiable {
    public var id: String { "\(position)-\(label)" }

    public let position: SwingPosition
    public let label: String
    public let status: FeedbackStatus
    public let detail: String
    public let drillIds: [String]?
}

public struct SwingThought: Codable, Equatable {
    public let position: SwingPosition
    public let label: String
    public let status: FeedbackStatus
    public let thought: String
    public let reason: String
    public let metricKey: String
}

public struct SwingAIPriorityFix: Codable, Equatable, Identifiable {
    public var id: String { "\(phase ?? "general")-\(title)" }

    public let title: String
    public let phase: String?
    public let observation: String
    public let whyItMatters: String
    public let cue: String

    public init(
        title: String,
        phase: String? = nil,
        observation: String,
        whyItMatters: String,
        cue: String
    ) {
        self.title = title
        self.phase = phase
        self.observation = observation
        self.whyItMatters = whyItMatters
        self.cue = cue
    }
}

public struct SwingAIPracticeDrill: Codable, Equatable, Identifiable {
    public var id: String { "\(title)-\(reps)" }

    public let title: String
    public let reps: String
    public let focus: String

    public init(title: String, reps: String, focus: String) {
        self.title = title
        self.reps = reps
        self.focus = focus
    }
}

public struct SwingAIFeedback: Codable, Equatable, Identifiable {
    public let id: String
    public let provider: String
    public let model: String
    public let generatedAt: String
    public let club: String
    public let sourceAnalysisVersion: Int?
    public let coachScore: Double
    public let headline: String
    public let summary: String
    public let confidence: String
    public let strengths: [String]
    public let priorityFixes: [SwingAIPriorityFix]
    public let practicePlan: [SwingAIPracticeDrill]
    public let limitations: [String]

    public init(
        id: String,
        provider: String,
        model: String,
        generatedAt: String,
        club: String,
        sourceAnalysisVersion: Int? = nil,
        coachScore: Double,
        headline: String,
        summary: String,
        confidence: String,
        strengths: [String],
        priorityFixes: [SwingAIPriorityFix],
        practicePlan: [SwingAIPracticeDrill],
        limitations: [String]
    ) {
        self.id = id
        self.provider = provider
        self.model = model
        self.generatedAt = generatedAt
        self.club = club
        self.sourceAnalysisVersion = sourceAnalysisVersion
        self.coachScore = coachScore
        self.headline = headline
        self.summary = summary
        self.confidence = confidence
        self.strengths = strengths
        self.priorityFixes = priorityFixes
        self.practicePlan = practicePlan
        self.limitations = limitations
    }
}

public struct AnalysisQuality: Codable, Equatable {
    public let status: String
    public let cameraAngle: String
    public let captureConfidence: Double
    public let scoreConfidence: Double
    public let poseCoverage: Double?
    public let bodyVisibility: Double?
    public let golferScale: Double?
    public let framing: Double?
    public let warnings: [String]
}

public extension AnalysisQuality {
    var cameraAngleDisplayName: String {
        Self.cameraAngleDisplayName(for: cameraAngle)
    }

    static func cameraAngleDisplayName(for angle: String) -> String {
        switch angle {
        case "down_the_line":
            "Down-the-line"
        case "face_on":
            "Face-on"
        case "side_profile":
            "Side-profile"
        default:
            "Unusual angle"
        }
    }
}

public struct SwingAnalysisDebug: Codable, Equatable {
    public let contactSheetError: String?
    public let contactSheetPath: String?
    public let contactSheetUrl: String?
    public let impactFilmstripError: String?
    public let impactFilmstripPath: String?
    public let impactFilmstripUrl: String?
    public let impactWindowTimesMs: [ImpactWindowFrame]?
    public let positionTimesMs: [String: PositionDebugTime]?
    public let analysisQuality: AnalysisQuality?
    public let phaseConfidence: [String: Double]?
}

public struct ImpactWindowFrame: Codable, Equatable, Identifiable {
    public var id: String { "\(frame)-\(offset)" }

    public let analysis: Int
    public let frame: Int
    public let offset: Int
    public let selected: Bool
    public let source: Int
}

public struct PositionDebugTime: Codable, Equatable {
    public let analysis: Int
    public let source: Int
}

public struct SwingAnalysis: Codable, Equatable {
    public let id: String
    public let videoUri: String
    public let fps: Double
    public let durationMs: Int
    public let analysisVersion: Int?
    public let debug: SwingAnalysisDebug?
    public let videoWidth: Int?
    public let videoHeight: Int?
    public let trimStartMs: Int
    public let trimEndMs: Int
    public let frames: [SwingAnalysisFrame]
    public let positions: [String: Int]
    public let metrics: SwingMetrics
    public let feedback: [FeedbackItem]
    public let swingThought: SwingThought?
    public let analysisQuality: AnalysisQuality?
    public let phaseConfidence: [String: Double]?
    public let shotEstimate: ShotEstimate?
    public let rawSenseiScore: Double?
    public let senseiScore: Double
    public let swingPathDiagnosis: SwingPathDiagnosis?

    public func frameIndex(for position: SwingPosition) -> Int? {
        guard let index = positions[position.rawValue], frames.indices.contains(index) else {
            return nil
        }
        return index
    }

    public func frame(for position: SwingPosition) -> SwingAnalysisFrame? {
        guard let index = frameIndex(for: position) else { return nil }
        return frames[index]
    }
}
