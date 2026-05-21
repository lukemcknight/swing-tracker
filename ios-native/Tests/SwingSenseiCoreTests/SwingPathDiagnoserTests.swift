import XCTest
@testable import SwingSenseiCore

final class SwingPathDiagnoserTests: XCTestCase {
    // An over-the-top loop: clubhead travels up the right side of the frame and
    // comes down the left (the downswing tracking outside the backswing).
    private let overTheTopLoop: [(Double, Double)] = [
        (0.50, 0.92), (0.58, 0.78), (0.66, 0.60), (0.70, 0.40), (0.68, 0.22),
        (0.60, 0.12),
        (0.48, 0.16), (0.40, 0.34), (0.36, 0.54), (0.40, 0.72), (0.46, 0.88),
    ]

    // The mirror image (x -> 1 - x): up the left, down the right.
    private let inToOutLoop: [(Double, Double)] = [
        (0.50, 0.92), (0.42, 0.78), (0.34, 0.60), (0.30, 0.40), (0.32, 0.22),
        (0.40, 0.12),
        (0.52, 0.16), (0.60, 0.34), (0.64, 0.54), (0.60, 0.72), (0.54, 0.88),
    ]

    // On-plane: the downswing exactly retraces the backswing — encloses no area.
    private let onPlaneLoop: [(Double, Double)] = [
        (0.50, 0.92), (0.58, 0.78), (0.66, 0.60), (0.70, 0.40), (0.68, 0.22),
        (0.60, 0.12),
        (0.68, 0.22), (0.70, 0.40), (0.66, 0.60), (0.58, 0.78), (0.50, 0.92),
    ]

    private func points(_ raw: [(Double, Double)]) -> [SwingPoint] {
        raw.map { SwingPoint(x: $0.0, y: $0.1) }
    }

    func testOverTheTopLoopClassifiedOutToIn() {
        let diagnosis = SwingPathDiagnoser.diagnose(
            path: points(overTheTopLoop),
            isDownTheLine: true
        )

        XCTAssertEqual(diagnosis?.direction, .outToIn)
        XCTAssertEqual(diagnosis?.verdict, "Out-to-in path — the classic slice pattern.")
        XCTAssertNotNil(diagnosis?.cue)
        XCTAssertGreaterThan(diagnosis?.severity ?? 0, 0.05)
    }

    func testInToOutLoopClassifiedInToOut() {
        let diagnosis = SwingPathDiagnoser.diagnose(
            path: points(inToOutLoop),
            isDownTheLine: true
        )

        XCTAssertEqual(diagnosis?.direction, .inToOut)
        XCTAssertEqual(diagnosis?.verdict, "In-to-out path — promotes a draw, or a hook/push if exaggerated.")
        XCTAssertNil(diagnosis?.cue)
        XCTAssertGreaterThan(diagnosis?.severity ?? 0, 0.05)
    }

    func testOnPlaneLoopClassifiedNeutral() {
        let diagnosis = SwingPathDiagnoser.diagnose(
            path: points(onPlaneLoop),
            isDownTheLine: true
        )

        XCTAssertEqual(diagnosis?.direction, .neutral)
        XCTAssertEqual(diagnosis?.verdict, "On-plane path — no slice tendency here.")
        XCTAssertNil(diagnosis?.cue)
        XCTAssertLessThan(diagnosis?.severity ?? 1, 0.05)
    }

    func testNonDownTheLineCaptureReturnsNil() {
        let diagnosis = SwingPathDiagnoser.diagnose(
            path: points(overTheTopLoop),
            isDownTheLine: false
        )

        XCTAssertNil(diagnosis)
    }

    func testTooFewPointsReturnsNil() {
        let diagnosis = SwingPathDiagnoser.diagnose(
            path: points(Array(overTheTopLoop.prefix(5))),
            isDownTheLine: true
        )

        XCTAssertNil(diagnosis)
    }

    func testZeroAreaBoundingBoxReturnsNil() {
        let verticalLine = (0..<11).map { i in (0.5, 0.1 + Double(i) * 0.07) }
        let diagnosis = SwingPathDiagnoser.diagnose(
            path: points(verticalLine),
            isDownTheLine: true
        )

        XCTAssertNil(diagnosis)
    }

    func testImbalancedBackswingAndDownswingReturnsNil() {
        // The top (minimum y) sits at index 1, leaving only one backswing point.
        let imbalanced: [(Double, Double)] = [
            (0.50, 0.92), (0.55, 0.12), (0.48, 0.30), (0.42, 0.45),
            (0.40, 0.60), (0.44, 0.72), (0.48, 0.84), (0.50, 0.90),
        ]
        let diagnosis = SwingPathDiagnoser.diagnose(
            path: points(imbalanced),
            isDownTheLine: true
        )

        XCTAssertNil(diagnosis)
    }

    func testDiagnosisCodableRoundTrip() throws {
        let original = SwingPathDiagnosis(
            direction: .outToIn,
            severity: 0.42,
            verdict: "Out-to-in path — the classic slice pattern.",
            cue: "Drop the club behind you."
        )

        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(SwingPathDiagnosis.self, from: data)

        XCTAssertEqual(decoded, original)
    }
}
