import Foundation

public enum RoundStatus: String, Codable, Equatable {
    case inProgress
    case complete

    public var label: String {
        switch self {
        case .inProgress:
            "In Progress"
        case .complete:
            "Complete"
        }
    }
}

public enum ArcadeLie: String, Codable, Equatable {
    case tee
    case fairway
    case rough
    case green
    case outOfBounds

    public var label: String {
        switch self {
        case .tee:
            "Tee"
        case .fairway:
            "Fairway"
        case .rough:
            "Rough"
        case .green:
            "Green"
        case .outOfBounds:
            "Out"
        }
    }
}

public enum ArcadeBallMode: String, Codable, CaseIterable, Identifiable, Equatable {
    case realBall
    case foamBall
    case noBall

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .realBall:
            "Real Ball"
        case .foamBall:
            "Foam Ball"
        case .noBall:
            "No Ball"
        }
    }

    public var shortLabel: String {
        switch self {
        case .realBall:
            "Real"
        case .foamBall:
            "Foam"
        case .noBall:
            "None"
        }
    }

    var distanceDispersionMultiplier: Double {
        switch self {
        case .realBall:
            1.0
        case .foamBall:
            1.35
        case .noBall:
            1.75
        }
    }

    var lateralDispersionMultiplier: Double {
        switch self {
        case .realBall:
            1.0
        case .foamBall:
            1.4
        case .noBall:
            1.8
        }
    }
}

public enum ArcadeShotConfidence: String, Codable, Equatable {
    case high
    case medium
    case low

    public init(_ value: String?) {
        switch value?.lowercased() {
        case "high":
            self = .high
        case "medium":
            self = .medium
        default:
            self = .low
        }
    }

    public var label: String {
        switch self {
        case .high:
            "High"
        case .medium:
            "Medium"
        case .low:
            "Low"
        }
    }

    var distanceDispersionYards: Double {
        switch self {
        case .high:
            8
        case .medium:
            16
        case .low:
            28
        }
    }

    var lateralDispersionYards: Double {
        switch self {
        case .high:
            7
        case .medium:
            14
        case .low:
            26
        }
    }
}

public struct ArcadePoint: Codable, Equatable {
    public var x: Double
    public var y: Double

    public init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }

    public func distance(to other: ArcadePoint) -> Double {
        let dx = x - other.x
        let dy = y - other.y
        return sqrt(dx * dx + dy * dy)
    }
}

public struct ArcadeHoleDefinition: Codable, Equatable, Identifiable {
    public var id: String
    public var number: Int
    public var par: Int
    public var lengthYards: Double
    public var fairwayWidthYards: Double
    public var greenRadiusYards: Double
    public var outOfBoundsWidthYards: Double

    public init(
        id: String,
        number: Int,
        par: Int,
        lengthYards: Double,
        fairwayWidthYards: Double,
        greenRadiusYards: Double,
        outOfBoundsWidthYards: Double
    ) {
        self.id = id
        self.number = number
        self.par = par
        self.lengthYards = lengthYards
        self.fairwayWidthYards = fairwayWidthYards
        self.greenRadiusYards = greenRadiusYards
        self.outOfBoundsWidthYards = outOfBoundsWidthYards
    }
}

public struct ArcadeCourseDefinition: Codable, Equatable, Identifiable {
    public var id: String
    public var name: String
    public var holes: [ArcadeHoleDefinition]

    public init(id: String, name: String, holes: [ArcadeHoleDefinition]) {
        self.id = id
        self.name = name
        self.holes = holes
    }
}

public enum ArcadeCourseLibrary {
    public static let starterCourse = ArcadeCourseDefinition(
        id: "starter-3",
        name: "Range Links",
        holes: [
            ArcadeHoleDefinition(
                id: "starter-3-1",
                number: 1,
                par: 3,
                lengthYards: 148,
                fairwayWidthYards: 38,
                greenRadiusYards: 17,
                outOfBoundsWidthYards: 96
            ),
            ArcadeHoleDefinition(
                id: "starter-3-2",
                number: 2,
                par: 4,
                lengthYards: 322,
                fairwayWidthYards: 44,
                greenRadiusYards: 18,
                outOfBoundsWidthYards: 108
            ),
            ArcadeHoleDefinition(
                id: "starter-3-3",
                number: 3,
                par: 5,
                lengthYards: 472,
                fairwayWidthYards: 50,
                greenRadiusYards: 20,
                outOfBoundsWidthYards: 124
            ),
        ]
    )
}

public struct RoundShot: Codable, Equatable, Identifiable {
    public var id: String
    public var createdAt: Date
    public var holeNumber: Int
    public var club: String
    public var ballMode: ArcadeBallMode
    public var swingID: String
    public var shotEstimate: ShotEstimate?
    public var startPosition: ArcadePoint
    public var endPosition: ArcadePoint
    public var carryYards: Double
    public var lateralMissYards: Double
    public var distanceDispersionYards: Double
    public var lateralDispersionYards: Double
    public var lie: ArcadeLie
    public var confidence: String
    public var autoPutts: Int
    public var penaltyStrokes: Int

    private enum CodingKeys: String, CodingKey {
        case id
        case createdAt
        case holeNumber
        case club
        case ballMode
        case swingID
        case shotEstimate
        case startPosition
        case endPosition
        case carryYards
        case lateralMissYards
        case distanceDispersionYards
        case lateralDispersionYards
        case lie
        case confidence
        case autoPutts
        case penaltyStrokes
    }

    public init(
        id: String = "shot-\(UUID().uuidString)",
        createdAt: Date = Date(),
        holeNumber: Int,
        club: String,
        ballMode: ArcadeBallMode = .realBall,
        swingID: String,
        shotEstimate: ShotEstimate?,
        startPosition: ArcadePoint,
        endPosition: ArcadePoint,
        carryYards: Double,
        lateralMissYards: Double,
        distanceDispersionYards: Double,
        lateralDispersionYards: Double,
        lie: ArcadeLie,
        confidence: String,
        autoPutts: Int = 0,
        penaltyStrokes: Int = 0
    ) {
        self.id = id
        self.createdAt = createdAt
        self.holeNumber = holeNumber
        self.club = club
        self.ballMode = ballMode
        self.swingID = swingID
        self.shotEstimate = shotEstimate
        self.startPosition = startPosition
        self.endPosition = endPosition
        self.carryYards = carryYards
        self.lateralMissYards = lateralMissYards
        self.distanceDispersionYards = distanceDispersionYards
        self.lateralDispersionYards = lateralDispersionYards
        self.lie = lie
        self.confidence = confidence
        self.autoPutts = autoPutts
        self.penaltyStrokes = penaltyStrokes
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        createdAt = try container.decode(Date.self, forKey: .createdAt)
        holeNumber = try container.decode(Int.self, forKey: .holeNumber)
        club = try container.decode(String.self, forKey: .club)
        ballMode = try container.decodeIfPresent(ArcadeBallMode.self, forKey: .ballMode) ?? .realBall
        swingID = try container.decode(String.self, forKey: .swingID)
        shotEstimate = try container.decodeIfPresent(ShotEstimate.self, forKey: .shotEstimate)
        startPosition = try container.decode(ArcadePoint.self, forKey: .startPosition)
        endPosition = try container.decode(ArcadePoint.self, forKey: .endPosition)
        carryYards = try container.decode(Double.self, forKey: .carryYards)
        lateralMissYards = try container.decode(Double.self, forKey: .lateralMissYards)
        distanceDispersionYards = try container.decode(Double.self, forKey: .distanceDispersionYards)
        lateralDispersionYards = try container.decode(Double.self, forKey: .lateralDispersionYards)
        lie = try container.decode(ArcadeLie.self, forKey: .lie)
        confidence = try container.decode(String.self, forKey: .confidence)
        autoPutts = try container.decodeIfPresent(Int.self, forKey: .autoPutts) ?? 0
        penaltyStrokes = try container.decodeIfPresent(Int.self, forKey: .penaltyStrokes) ?? 0
    }

    public var strokesAdded: Int {
        1 + autoPutts + penaltyStrokes
    }
}

public struct HoleState: Codable, Equatable, Identifiable {
    public var id: String
    public var holeNumber: Int
    public var par: Int
    public var lengthYards: Double
    public var fairwayWidthYards: Double
    public var greenCenter: ArcadePoint
    public var greenRadiusYards: Double
    public var outOfBoundsWidthYards: Double
    public var ballPosition: ArcadePoint
    public var lie: ArcadeLie
    public var strokes: Int
    public var completed: Bool
    public var shots: [RoundShot]

    public init(
        id: String,
        holeNumber: Int,
        par: Int,
        lengthYards: Double,
        fairwayWidthYards: Double,
        greenCenter: ArcadePoint,
        greenRadiusYards: Double,
        outOfBoundsWidthYards: Double,
        ballPosition: ArcadePoint = ArcadePoint(x: 0, y: 0),
        lie: ArcadeLie = .tee,
        strokes: Int = 0,
        completed: Bool = false,
        shots: [RoundShot] = []
    ) {
        self.id = id
        self.holeNumber = holeNumber
        self.par = par
        self.lengthYards = lengthYards
        self.fairwayWidthYards = fairwayWidthYards
        self.greenCenter = greenCenter
        self.greenRadiusYards = greenRadiusYards
        self.outOfBoundsWidthYards = outOfBoundsWidthYards
        self.ballPosition = ballPosition
        self.lie = lie
        self.strokes = strokes
        self.completed = completed
        self.shots = shots
    }

    public init(definition: ArcadeHoleDefinition) {
        self.init(
            id: definition.id,
            holeNumber: definition.number,
            par: definition.par,
            lengthYards: definition.lengthYards,
            fairwayWidthYards: definition.fairwayWidthYards,
            greenCenter: ArcadePoint(x: 0, y: definition.lengthYards),
            greenRadiusYards: definition.greenRadiusYards,
            outOfBoundsWidthYards: definition.outOfBoundsWidthYards
        )
    }

    public var distanceRemainingYards: Double {
        max(0, ballPosition.distance(to: greenCenter))
    }

    public var relativeScore: Int {
        strokes - par
    }
}

public struct RoundRecord: Codable, Equatable, Identifiable {
    public var id: String
    public var startedAt: Date
    public var endedAt: Date?
    public var courseID: String
    public var courseName: String
    public var holes: [HoleState]
    public var totalStrokes: Int
    public var status: RoundStatus

    private enum CodingKeys: String, CodingKey {
        case id
        case startedAt
        case endedAt
        case courseID
        case courseName
        case holes
        case totalStrokes
        case status
    }

    public init(
        id: String = "round-\(UUID().uuidString)",
        startedAt: Date = Date(),
        endedAt: Date? = nil,
        courseID: String,
        courseName: String,
        holes: [HoleState],
        totalStrokes: Int? = nil,
        status: RoundStatus = .inProgress
    ) {
        self.id = id
        self.startedAt = startedAt
        self.endedAt = endedAt
        self.courseID = courseID
        self.courseName = courseName
        self.holes = holes
        self.totalStrokes = totalStrokes ?? holes.reduce(0) { $0 + $1.strokes }
        self.status = status
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        startedAt = try container.decode(Date.self, forKey: .startedAt)
        endedAt = try container.decodeIfPresent(Date.self, forKey: .endedAt)
        courseID = try container.decode(String.self, forKey: .courseID)
        courseName = try container.decodeIfPresent(String.self, forKey: .courseName) ?? "Range Links"
        holes = try container.decode([HoleState].self, forKey: .holes)
        totalStrokes = try container.decodeIfPresent(Int.self, forKey: .totalStrokes)
            ?? holes.reduce(0) { $0 + $1.strokes }
        status = try container.decodeIfPresent(RoundStatus.self, forKey: .status) ?? .inProgress
    }

    public static func starter(
        id: String = "round-\(UUID().uuidString)",
        startedAt: Date = Date(),
        course: ArcadeCourseDefinition = ArcadeCourseLibrary.starterCourse
    ) -> RoundRecord {
        RoundRecord(
            id: id,
            startedAt: startedAt,
            courseID: course.id,
            courseName: course.name,
            holes: course.holes.map(HoleState.init(definition:))
        )
    }

    public var currentHoleIndex: Int? {
        holes.firstIndex { !$0.completed }
    }

    public var currentHole: HoleState? {
        guard let currentHoleIndex else { return nil }
        return holes[currentHoleIndex]
    }

    public var totalPar: Int {
        holes.reduce(0) { $0 + $1.par }
    }

    public var completedHoles: Int {
        holes.filter(\.completed).count
    }

    public var scoreToPar: Int {
        totalStrokes - holes.prefix(completedHoles).reduce(0) { $0 + $1.par }
    }

    public mutating func replaceHole(_ hole: HoleState, endedAt date: Date = Date()) {
        guard let index = holes.firstIndex(where: { $0.id == hole.id }) else { return }
        holes[index] = hole
        totalStrokes = holes.reduce(0) { $0 + $1.strokes }
        if holes.allSatisfy(\.completed) {
            status = .complete
            endedAt = date
        }
    }

    public mutating func undoLastShot() -> Bool {
        let holeIndex = holes.firstIndex { !$0.shots.isEmpty && ($0.completed || $0.id == currentHole?.id) }
            ?? holes.lastIndex { !$0.shots.isEmpty }
        guard let index = holeIndex else { return false }
        var hole = holes[index]
        guard let removed = hole.shots.popLast() else { return false }
        hole.strokes = max(0, hole.strokes - removed.strokesAdded)
        hole.ballPosition = removed.startPosition
        hole.lie = hole.shots.last?.lie ?? .tee
        hole.completed = false
        holes[index] = hole
        totalStrokes = holes.reduce(0) { $0 + $1.strokes }
        if status == .complete {
            status = .inProgress
            endedAt = nil
        }
        return true
    }

    public mutating func restartHoles() {
        let definitions = holes.map { existing in
            ArcadeHoleDefinition(
                id: existing.id,
                number: existing.holeNumber,
                par: existing.par,
                lengthYards: existing.lengthYards,
                fairwayWidthYards: existing.fairwayWidthYards,
                greenRadiusYards: existing.greenRadiusYards,
                outOfBoundsWidthYards: existing.outOfBoundsWidthYards
            )
        }
        holes = definitions.map(HoleState.init(definition:))
        totalStrokes = 0
        status = .inProgress
        endedAt = nil
        startedAt = Date()
    }

    public mutating func abandon(at date: Date = Date()) {
        status = .complete
        endedAt = date
    }
}

public struct ArcadeShotResult: Codable, Equatable {
    public var shot: RoundShot
    public var updatedHole: HoleState

    public init(shot: RoundShot, updatedHole: HoleState) {
        self.shot = shot
        self.updatedHole = updatedHole
    }
}

public struct ArcadeShotEngine {
    public init() {}

    public func resolveShot(
        hole: HoleState,
        club clubValue: String,
        ballMode: ArcadeBallMode = .realBall,
        swingID: String,
        shotEstimate: ShotEstimate?,
        createdAt: Date = Date(),
        shotID: String = "shot-\(UUID().uuidString)"
    ) -> ArcadeShotResult {
        let club = GolfClub.canonical(clubValue) ?? .other
        let confidence = resolvedConfidence(for: shotEstimate, ballMode: ballMode)
        let carry = resolvedCarryYards(
            club: club,
            shotEstimate: shotEstimate,
            confidence: confidence,
            ballMode: ballMode
        )
        let boundedCarry = min(max(18, carry), max(24, hole.distanceRemainingYards + 36))
        let lateralMiss = resolvedLateralMissYards(
            hole: hole,
            shotEstimate: shotEstimate,
            confidence: confidence,
            ballMode: ballMode,
            swingID: swingID
        )
        let distanceDispersion = confidence.distanceDispersionYards * ballMode.distanceDispersionMultiplier
        let lateralDispersion = confidence.lateralDispersionYards * ballMode.lateralDispersionMultiplier

        var landing = ArcadePoint(
            x: hole.ballPosition.x + lateralMiss,
            y: min(hole.lengthYards + 36, hole.ballPosition.y + boundedCarry)
        )
        var lie = classifyLie(for: landing, hole: hole)
        var penaltyStrokes = 0

        if lie == .outOfBounds {
            penaltyStrokes = 1
            let side = landing.x < 0 ? -1.0 : 1.0
            landing.x = side * max(0, hole.outOfBoundsWidthYards / 2 - 8)
            lie = .rough
        }

        var autoPutts = 0
        if landing.distance(to: hole.greenCenter) <= hole.greenRadiusYards {
            lie = .green
            autoPutts = Self.autoPutts(distanceToPinYards: landing.distance(to: hole.greenCenter))
        }

        let shot = RoundShot(
            id: shotID,
            createdAt: createdAt,
            holeNumber: hole.holeNumber,
            club: club.rawValue,
            ballMode: ballMode,
            swingID: swingID,
            shotEstimate: shotEstimate,
            startPosition: hole.ballPosition,
            endPosition: landing,
            carryYards: boundedCarry,
            lateralMissYards: lateralMiss,
            distanceDispersionYards: distanceDispersion,
            lateralDispersionYards: lateralDispersion,
            lie: lie,
            confidence: confidence.rawValue,
            autoPutts: autoPutts,
            penaltyStrokes: penaltyStrokes
        )

        var updatedHole = hole
        updatedHole.ballPosition = landing
        updatedHole.lie = lie
        updatedHole.strokes += shot.strokesAdded
        updatedHole.completed = autoPutts > 0
        updatedHole.shots.append(shot)

        return ArcadeShotResult(shot: shot, updatedHole: updatedHole)
    }

    public static func autoPutts(distanceToPinYards: Double) -> Int {
        if distanceToPinYards <= 2 {
            return 1
        }
        if distanceToPinYards <= 10 {
            return 2
        }
        return 3
    }

    private func resolvedConfidence(
        for shotEstimate: ShotEstimate?,
        ballMode: ArcadeBallMode
    ) -> ArcadeShotConfidence {
        if ballMode == .noBall {
            return .low
        }
        guard let shotEstimate else { return .low }
        if ballMode == .foamBall {
            guard shotEstimate.clubSpeedMphRange != nil else { return .low }
            return minConfidence(ArcadeShotConfidence(shotEstimate.confidence), .medium)
        }
        if shotEstimate.ballSpeedMphRange == nil, shotEstimate.clubSpeedMphRange == nil {
            return .low
        }
        return ArcadeShotConfidence(shotEstimate.confidence)
    }

    private func resolvedCarryYards(
        club: GolfClub,
        shotEstimate: ShotEstimate?,
        confidence: ArcadeShotConfidence,
        ballMode: ArcadeBallMode
    ) -> Double {
        guard let shotEstimate else {
            return club.arcadeDefaultCarryYards * 0.86
        }

        if ballMode == .realBall, let ballSpeed = shotEstimate.ballSpeedMphRange?.midpoint {
            return carryYards(club: club, ballSpeedMph: ballSpeed, confidence: confidence)
        }

        if let clubSpeed = shotEstimate.clubSpeedMphRange?.midpoint {
            return carryYards(
                club: club,
                ballSpeedMph: clubSpeed * club.arcadeSmashFactor,
                confidence: ballMode == .noBall ? .low : minConfidence(confidence, .medium)
            )
        }

        let fallbackMultiplier: Double
        switch ballMode {
        case .realBall:
            fallbackMultiplier = 0.86
        case .foamBall:
            fallbackMultiplier = 0.82
        case .noBall:
            fallbackMultiplier = 0.76
        }
        return club.arcadeDefaultCarryYards * fallbackMultiplier
    }

    private func carryYards(club: GolfClub, ballSpeedMph: Double, confidence: ArcadeShotConfidence) -> Double {
        let speedRatio = max(0.50, min(1.30, ballSpeedMph / club.arcadeReferenceBallSpeedMph))
        let confidenceMultiplier: Double
        switch confidence {
        case .high:
            confidenceMultiplier = 1.0
        case .medium:
            confidenceMultiplier = 0.97
        case .low:
            confidenceMultiplier = 0.92
        }
        return club.arcadeDefaultCarryYards * pow(speedRatio, 1.16) * confidenceMultiplier
    }

    private func resolvedLateralMissYards(
        hole: HoleState,
        shotEstimate: ShotEstimate?,
        confidence: ArcadeShotConfidence,
        ballMode: ArcadeBallMode,
        swingID: String
    ) -> Double {
        if ballMode == .noBall {
            return 0
        }
        let zone = shotEstimate?.landingZone.lowercased() ?? "unknown"
        let direction: Double
        switch zone {
        case "left":
            direction = -1
        case "right":
            direction = 1
        default:
            direction = 0
        }

        let modeMultiplier = ballMode == .foamBall ? 0.85 : 1.0
        let jitter = Self.deterministicUnitValue(for: swingID)
        let dispersion = confidence.lateralDispersionYards

        if direction == 0 {
            let driftRange = dispersion * (1 - confidence.driftDampening) * 0.6
            let centerDrift = (jitter - 0.5) * 2 * driftRange
            return centerDrift * modeMultiplier
        }

        let baseMagnitude = hole.fairwayWidthYards * 0.28 + dispersion * 0.25
        let jitterMagnitude = (jitter - 0.5) * 2 * dispersion * 0.3
        let signedMagnitude = max(0, baseMagnitude + jitterMagnitude)
        return direction * signedMagnitude * modeMultiplier
    }

    static func deterministicUnitValue(for seed: String) -> Double {
        var hash: UInt64 = 0xcbf29ce484222325
        for scalar in seed.unicodeScalars {
            hash ^= UInt64(scalar.value)
            hash = hash &* 0x100000001b3
        }
        let mantissa = hash & 0x000fffffffffffff
        return Double(mantissa) / Double(0x000fffffffffffff)
    }

    private func classifyLie(for point: ArcadePoint, hole: HoleState) -> ArcadeLie {
        if point.distance(to: hole.greenCenter) <= hole.greenRadiusYards {
            return .green
        }
        if abs(point.x) > hole.outOfBoundsWidthYards / 2 {
            return .outOfBounds
        }
        if abs(point.x) <= hole.fairwayWidthYards / 2 {
            return .fairway
        }
        return .rough
    }

    private func minConfidence(
        _ lhs: ArcadeShotConfidence,
        _ rhs: ArcadeShotConfidence
    ) -> ArcadeShotConfidence {
        if lhs.rank <= rhs.rank {
            return lhs
        }
        return rhs
    }
}

private extension ArcadeShotConfidence {
    var rank: Int {
        switch self {
        case .low:
            0
        case .medium:
            1
        case .high:
            2
        }
    }

    var driftDampening: Double {
        switch self {
        case .high:
            1.0
        case .medium:
            0.7
        case .low:
            0.4
        }
    }
}

public extension SpeedRange {
    var midpoint: Double {
        (min + max) / 2
    }
}

public extension GolfClub {
    var arcadeDefaultCarryYards: Double {
        switch self {
        case .driver:
            230
        case .threeWood:
            210
        case .fiveWood:
            195
        case .threeHybrid:
            185
        case .fourHybrid:
            175
        case .fiveHybrid:
            165
        case .threeIron:
            180
        case .fourIron:
            170
        case .fiveIron:
            160
        case .sixIron:
            150
        case .sevenIron:
            140
        case .eightIron:
            128
        case .nineIron:
            116
        case .pitchingWedge:
            102
        case .approachWedge:
            88
        case .sandWedge:
            72
        case .lobWedge:
            58
        case .other:
            118
        }
    }

    var arcadeReferenceBallSpeedMph: Double {
        switch self {
        case .driver:
            140
        case .threeWood:
            130
        case .fiveWood:
            123
        case .threeHybrid:
            118
        case .fourHybrid:
            113
        case .fiveHybrid:
            108
        case .threeIron:
            116
        case .fourIron:
            111
        case .fiveIron:
            106
        case .sixIron:
            101
        case .sevenIron:
            96
        case .eightIron:
            91
        case .nineIron:
            86
        case .pitchingWedge:
            80
        case .approachWedge:
            74
        case .sandWedge:
            66
        case .lobWedge:
            58
        case .other:
            92
        }
    }

    var arcadeSmashFactor: Double {
        switch self {
        case .driver:
            1.45
        case .threeWood, .fiveWood:
            1.40
        case .threeHybrid, .fourHybrid, .fiveHybrid:
            1.36
        case .threeIron, .fourIron, .fiveIron:
            1.33
        case .sixIron, .sevenIron, .eightIron, .nineIron:
            1.29
        case .pitchingWedge, .approachWedge:
            1.22
        case .sandWedge, .lobWedge:
            1.12
        case .other:
            1.25
        }
    }
}

public enum RoundRepositoryError: Error, LocalizedError {
    case roundNotFound(String)

    public var errorDescription: String? {
        switch self {
        case let .roundNotFound(id):
            "Could not find round \(id)."
        }
    }
}

public final class RoundRepository {
    public let rootDirectory: URL
    public let metadataURL: URL

    private let fileManager: FileManager
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    public init(
        rootDirectory: URL? = nil,
        fileManager: FileManager = .default
    ) {
        self.fileManager = fileManager

        let baseDirectory = rootDirectory ?? fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("SwingSensei", isDirectory: true)
        self.rootDirectory = baseDirectory
        self.metadataURL = baseDirectory.appendingPathComponent("rounds.json")

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        self.encoder = encoder

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        self.decoder = decoder
    }

    public func load() throws -> [RoundRecord] {
        try ensureDirectoryExists()
        guard fileManager.fileExists(atPath: metadataURL.path) else {
            return []
        }

        let data = try Data(contentsOf: metadataURL)
        return try decoder.decode([RoundRecord].self, from: data)
            .sorted { $0.startedAt > $1.startedAt }
    }

    public func save(_ rounds: [RoundRecord]) throws {
        try ensureDirectoryExists()
        let data = try encoder.encode(rounds.sorted { $0.startedAt > $1.startedAt })
        try data.write(to: metadataURL, options: [.atomic])
    }

    public func upserting(_ round: RoundRecord, into rounds: [RoundRecord]) -> [RoundRecord] {
        [round] + rounds.filter { $0.id != round.id }
    }

    private func ensureDirectoryExists() throws {
        try fileManager.createDirectory(at: rootDirectory, withIntermediateDirectories: true)
    }
}
