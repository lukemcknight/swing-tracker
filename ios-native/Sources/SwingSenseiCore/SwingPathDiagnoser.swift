import Foundation

/// Plain-language verdict on the swing path, derived purely from the detected
/// clubhead loop. See docs/superpowers/specs/2026-05-20-slice-diagnosis-design.md.
public struct SwingPathDiagnosis: Codable, Equatable {
    public enum PathDirection: String, Codable {
        case outToIn
        case neutral
        case inToOut
    }

    /// Loop direction: out-to-in (the classic slice pattern), neutral, or in-to-out.
    public let direction: PathDirection
    /// Normalized |signed loop area| — magnitude of the path fault. Retained for
    /// future use; the MVP only acts on `direction`.
    public let severity: Double
    /// Plain-language headline.
    public let verdict: String
    /// Corrective cue; `nil` when there is nothing to fix.
    public let cue: String?

    public init(direction: PathDirection, severity: Double, verdict: String, cue: String?) {
        self.direction = direction
        self.severity = severity
        self.verdict = verdict
        self.cue = cue
    }
}

/// Pure geometry on the cleaned clubhead path. No platform dependencies; fully
/// unit-testable. Total: any input yields either a valid diagnosis or `nil`.
public enum SwingPathDiagnoser {
    /// Minimum detected points for a meaningful loop.
    static let minimumPathPoints = 8
    /// Minimum points required on each side of the top of the swing.
    static let minimumTrackPoints = 3

    /// Sign convention for the shoelace signed area.
    ///
    /// In image coordinates (y increases downward), a visually-clockwise loop has
    /// a positive signed area. Our working model: a down-the-line, right-handed
    /// over-the-top swing traces the clubhead up the right side and down the
    /// left (the downswing outside the backswing) — a counter-clockwise loop, so
    /// a NEGATIVE signed area. Hence `false`.
    ///
    /// This is calibrated against a real clip in Task 6 of the implementation
    /// plan. If a known out-to-in clip classifies as `.inToOut`, flip this
    /// constant and swap the expected directions in SwingPathDiagnoserTests.
    static let outToInSignIsPositive = false

    public static func diagnose(
        path: [SwingPoint],
        isDownTheLine: Bool,
        neutralThreshold: Double = 0.05
    ) -> SwingPathDiagnosis? {
        // Gate: swing path is only readable from a down-the-line capture.
        guard isDownTheLine else { return nil }

        // Gate: too few detected points overall.
        guard path.count >= minimumPathPoints else { return nil }

        // Gate: degenerate bounding box.
        let xs = path.map(\.x)
        let ys = path.map(\.y)
        guard
            let minX = xs.min(), let maxX = xs.max(),
            let minY = ys.min(), let maxY = ys.max()
        else { return nil }
        let boundingBoxArea = (maxX - minX) * (maxY - minY)
        guard boundingBoxArea > 0 else { return nil }

        // Gate: both the backswing and downswing tracks must be present. The top
        // of the swing is the highest point on screen (minimum y).
        guard let topIndex = path.indices.min(by: { path[$0].y < path[$1].y }) else {
            return nil
        }
        let backswingPointCount = topIndex
        let downswingPointCount = path.count - topIndex - 1
        guard
            backswingPointCount >= minimumTrackPoints,
            downswingPointCount >= minimumTrackPoints
        else { return nil }

        // Signed loop area (shoelace formula, polygon implicitly closed last->first).
        var doubleArea = 0.0
        for index in path.indices {
            let current = path[index]
            let next = path[(index + 1) % path.count]
            doubleArea += current.x * next.y - next.x * current.y
        }
        let signedArea = doubleArea / 2.0
        let severity = abs(signedArea) / boundingBoxArea

        let direction: SwingPathDiagnosis.PathDirection
        if severity < neutralThreshold {
            direction = .neutral
        } else {
            let isOutToInLoop = outToInSignIsPositive ? (signedArea > 0) : (signedArea < 0)
            direction = isOutToInLoop ? .outToIn : .inToOut
        }

        return SwingPathDiagnosis(
            direction: direction,
            severity: severity,
            verdict: verdict(for: direction),
            cue: cue(for: direction)
        )
    }

    private static func verdict(for direction: SwingPathDiagnosis.PathDirection) -> String {
        switch direction {
        case .outToIn:
            return "Out-to-in path — the classic slice pattern."
        case .neutral:
            return "On-plane path — no slice tendency here."
        case .inToOut:
            return "In-to-out path — promotes a draw, or a hook/push if exaggerated."
        }
    }

    private static func cue(for direction: SwingPathDiagnosis.PathDirection) -> String? {
        switch direction {
        case .outToIn:
            return "Feel the club drop behind you as you start down, rather than throwing it out toward the ball."
        case .neutral, .inToOut:
            return nil
        }
    }
}
