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
    public static func diagnose(
        path: [SwingPoint],
        isDownTheLine: Bool,
        neutralThreshold: Double = 0.05
    ) -> SwingPathDiagnosis? {
        nil
    }
}
