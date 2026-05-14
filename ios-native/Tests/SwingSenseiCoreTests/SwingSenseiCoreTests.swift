import CoreGraphics
import XCTest
@testable import SwingSenseiCore

final class SwingSenseiCoreTests: XCTestCase {
    func testDecodesSwingAnalysisContract() throws {
        let analysis = try makeFixtureAnalysis()

        XCTAssertEqual(analysis.id, "fixture")
        XCTAssertEqual(analysis.analysisVersion, 20)
        XCTAssertEqual(analysis.positions["impact"], 2)
        XCTAssertNil(analysis.frames[0].clubHead)
        XCTAssertEqual(analysis.frames[2].clubHead, SwingPoint(x: 0.62, y: 0.78))
        XCTAssertEqual(analysis.swingThought?.thought, "Keep your chest over the ball.")
        XCTAssertEqual(analysis.analysisQuality?.status, "ok")
        XCTAssertEqual(analysis.phaseConfidence?["impact"], 0.92)
        XCTAssertEqual(analysis.debug?.contactSheetUrl, "/debug/fixture_phases_v20.jpg")
        XCTAssertEqual(analysis.debug?.impactFilmstripUrl, "/debug/fixture_impact_v20.jpg")
    }

    func testDecodesSideProfileAnalysisQualityDisplayName() throws {
        let analysis = try makeFixtureAnalysis(cameraAngle: "side_profile")

        XCTAssertEqual(analysis.analysisQuality?.cameraAngle, "side_profile")
        XCTAssertEqual(analysis.analysisQuality?.cameraAngleDisplayName, "Side-profile")
        XCTAssertEqual(analysis.debug?.analysisQuality?.cameraAngleDisplayName, "Side-profile")
    }

    func testPhaseMarkersExtractTimesFromPositionsAndFrames() throws {
        let analysis = try makeFixtureAnalysis()

        let markers = analysis.phaseMarkers()
        XCTAssertEqual(markers.map(\.position), [.address, .takeaway, .top, .impact, .finish])
        XCTAssertEqual(analysis.marker(for: .impact)?.timeMs, 1000)
        XCTAssertEqual(analysis.marker(for: .finish)?.progress, 0.625)
    }

    func testNearestFrameLookupChoosesClosestTimestamp() throws {
        let analysis = try makeFixtureAnalysis()

        XCTAssertEqual(analysis.nearestFrameIndex(to: 0), 0)
        XCTAssertEqual(analysis.nearestFrameIndex(to: 860), 2)
        XCTAssertEqual(analysis.nearestFrameIndex(to: 1350), 3)
        XCTAssertEqual(analysis.closestPosition(to: 990), .impact)
    }

    func testStableClubPathSegmentsPreserveSmoothPath() {
        let frames = makeClubPathFrames(points: [
            SwingPoint(x: 0.62, y: 0.82),
            SwingPoint(x: 0.64, y: 0.81),
            SwingPoint(x: 0.66, y: 0.80),
            SwingPoint(x: 0.68, y: 0.79),
            SwingPoint(x: 0.70, y: 0.78),
        ])

        let segments = ClubPathStabilizer.stableSegments(from: frames)

        XCTAssertEqual(segments.count, 1)
        XCTAssertEqual(segments[0].samples.map(\.frameIndex), [0, 1, 2, 3, 4])
    }

    func testStableClubPathSegmentsRejectJumpyExistingAnalysisData() {
        let frames = makeClubPathFrames(points: [
            SwingPoint(x: 0.62, y: 0.82),
            SwingPoint(x: 0.98, y: 0.54),
            SwingPoint(x: 0.20, y: 0.86),
            SwingPoint(x: 0.95, y: 0.50),
            SwingPoint(x: 0.22, y: 0.88),
            SwingPoint(x: 0.90, y: 0.48),
        ])

        let segments = ClubPathStabilizer.stableSegments(from: frames)

        XCTAssertTrue(segments.isEmpty)
    }

    func testStableClubPathSegmentsBreakAcrossGaps() {
        let frames = makeClubPathFrames(points: [
            SwingPoint(x: 0.62, y: 0.82),
            SwingPoint(x: 0.64, y: 0.81),
            SwingPoint(x: 0.66, y: 0.80),
            SwingPoint(x: 0.68, y: 0.79),
            nil,
            SwingPoint(x: 0.70, y: 0.78),
            SwingPoint(x: 0.72, y: 0.77),
            SwingPoint(x: 0.74, y: 0.76),
            SwingPoint(x: 0.76, y: 0.75),
        ])

        let segments = ClubPathStabilizer.stableSegments(from: frames)

        XCTAssertEqual(segments.count, 2)
        XCTAssertEqual(segments[0].samples.map(\.frameIndex), [0, 1, 2, 3])
        XCTAssertEqual(segments[1].samples.map(\.frameIndex), [5, 6, 7, 8])
    }

    func testPrimaryClubPathSegmentPreservesCleanPath() {
        let frames = makeClubPathFrames(points: [
            SwingPoint(x: 0.62, y: 0.82),
            SwingPoint(x: 0.64, y: 0.81),
            SwingPoint(x: 0.66, y: 0.80),
            SwingPoint(x: 0.68, y: 0.79),
            SwingPoint(x: 0.70, y: 0.78),
        ])

        let segment = ClubPathStabilizer.primarySegment(
            from: frames,
            positions: ["address": 0, "takeaway": 1, "top": 2, "impact": 4, "finish": 4]
        )

        XCTAssertEqual(segment?.samples.map(\.frameIndex), [0, 1, 2, 3, 4])
    }

    func testPrimaryClubPathSegmentKeepsThreePointRangeTrack() {
        let frames = makeClubPathFrames(points: [
            SwingPoint(x: 0.62, y: 0.82),
            SwingPoint(x: 0.65, y: 0.80),
            SwingPoint(x: 0.69, y: 0.77),
            nil,
            nil,
        ])

        let segment = ClubPathStabilizer.primarySegment(
            from: frames,
            positions: ["address": 0, "takeaway": 1, "top": 2, "impact": 4, "finish": 4]
        )

        XCTAssertEqual(segment?.samples.map(\.frameIndex), [0, 1, 2])
    }

    func testPrimaryClubPathSegmentKeepsPhaseAlignedRun() {
        let frames = makeClubPathFrames(points: [
            SwingPoint(x: 0.18, y: 0.64),
            SwingPoint(x: 0.20, y: 0.64),
            SwingPoint(x: 0.22, y: 0.65),
            SwingPoint(x: 0.24, y: 0.65),
            nil,
            SwingPoint(x: 0.62, y: 0.82),
            SwingPoint(x: 0.64, y: 0.81),
            SwingPoint(x: 0.66, y: 0.80),
            SwingPoint(x: 0.68, y: 0.79),
            SwingPoint(x: 0.70, y: 0.78),
        ])

        let segment = ClubPathStabilizer.primarySegment(
            from: frames,
            positions: ["address": 5, "takeaway": 6, "top": 7, "impact": 9, "finish": 9]
        )

        XCTAssertEqual(segment?.samples.map(\.frameIndex), [5, 6, 7, 8, 9])
    }

    func testPrimaryClubPathSegmentKeepsFullSwingWithLongFollowThrough() {
        let points = (0...60).map { index in
            let amount = Double(index) / 60
            return SwingPoint(
                x: 0.62 + sin(amount * Double.pi * 0.85) * 0.20,
                y: 0.82 - sin(amount * Double.pi) * 0.52
            )
        }
        let frames = makeClubPathFrames(points: points)

        let segment = ClubPathStabilizer.primarySegment(
            from: frames,
            positions: ["address": 0, "takeaway": 5, "top": 10, "impact": 14, "finish": 60]
        )

        XCTAssertEqual(segment?.samples.map(\.frameIndex), Array(0...60))
    }

    func testPrimaryClubPathSegmentRejectsAmbiguousCompetingRuns() {
        let frames = makeClubPathFrames(points: [
            SwingPoint(x: 0.62, y: 0.82),
            SwingPoint(x: 0.64, y: 0.81),
            SwingPoint(x: 0.66, y: 0.80),
            SwingPoint(x: 0.68, y: 0.79),
            nil,
            SwingPoint(x: 0.68, y: 0.79),
            SwingPoint(x: 0.70, y: 0.78),
            SwingPoint(x: 0.72, y: 0.77),
            SwingPoint(x: 0.74, y: 0.76),
        ])

        let segment = ClubPathStabilizer.primarySegment(
            from: frames,
            positions: ["address": 0, "takeaway": 1, "top": 4, "impact": 8, "finish": 8]
        )

        XCTAssertNil(segment)
    }

    func testPrimaryClubPathSegmentRejectsRunOutsideSwingWindow() {
        let frames = makeClubPathFrames(points: [
            nil,
            nil,
            nil,
            nil,
            nil,
            SwingPoint(x: 0.15, y: 0.03),
            SwingPoint(x: 0.15, y: 0.02),
            SwingPoint(x: 0.14, y: 0.01),
            SwingPoint(x: 0.14, y: 0.00),
            SwingPoint(x: 0.13, y: 0.00),
            SwingPoint(x: 0.13, y: 0.00),
        ])

        let segment = ClubPathStabilizer.primarySegment(
            from: frames,
            positions: ["address": 0, "takeaway": 1, "top": 2, "impact": 3, "finish": 4]
        )

        XCTAssertNil(segment)
    }

    func testPrimaryClubPathSegmentRejectsPostImpactOnlyRun() {
        let frames = makeClubPathFrames(points: [
            nil,
            nil,
            nil,
            nil,
            nil,
            nil,
            nil,
            SwingPoint(x: 0.72, y: 0.82),
            SwingPoint(x: 0.73, y: 0.75),
            SwingPoint(x: 0.74, y: 0.68),
            SwingPoint(x: 0.75, y: 0.61),
            SwingPoint(x: 0.76, y: 0.54),
        ])

        let segment = ClubPathStabilizer.primarySegment(
            from: frames,
            positions: ["address": 0, "takeaway": 2, "top": 4, "impact": 6, "finish": 11]
        )

        XCTAssertNil(segment)
    }

    func testAspectFitAndNormalizedPointMapping() {
        let rect = VideoGeometry.aspectFitRect(
            videoSize: CGSize(width: 1080, height: 1920),
            containerSize: CGSize(width: 430, height: 600)
        )

        XCTAssertEqual(rect.width, 337.5, accuracy: 0.001)
        XCTAssertEqual(rect.height, 600, accuracy: 0.001)
        XCTAssertEqual(rect.minX, 46.25, accuracy: 0.001)

        let point = VideoGeometry.mapNormalizedPoint(x: 0.5, y: 0.25, into: rect)
        XCTAssertEqual(point.x, 215, accuracy: 0.001)
        XCTAssertEqual(point.y, 150, accuracy: 0.001)
    }

    func testVideoCoordinateSpaceAppliesAssetRotationWhenAnalysisUsesRawDimensions() {
        let transform = CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 1080, ty: 0)
        let coordinateSpace = VideoCoordinateSpace.displayingAsset(
            analysisSize: CGSize(width: 1920, height: 1080),
            naturalSize: CGSize(width: 1920, height: 1080),
            preferredTransform: transform
        )

        XCTAssertEqual(coordinateSpace.displaySize.width, 1080, accuracy: 0.001)
        XCTAssertEqual(coordinateSpace.displaySize.height, 1920, accuracy: 0.001)

        let mapped = coordinateSpace.mapNormalizedPoint(x: 0.25, y: 0.5)
        XCTAssertEqual(mapped.x, 0.5, accuracy: 0.001)
        XCTAssertEqual(mapped.y, 0.25, accuracy: 0.001)
    }

    func testVideoCoordinateSpaceDoesNotDoubleRotateDisplayOrientedAnalysis() {
        let transform = CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 1080, ty: 0)
        let coordinateSpace = VideoCoordinateSpace.displayingAsset(
            analysisSize: CGSize(width: 1080, height: 1920),
            naturalSize: CGSize(width: 1920, height: 1080),
            preferredTransform: transform
        )

        XCTAssertEqual(coordinateSpace.displaySize.width, 1080, accuracy: 0.001)
        XCTAssertEqual(coordinateSpace.displaySize.height, 1920, accuracy: 0.001)

        let mapped = coordinateSpace.mapNormalizedPoint(x: 0.25, y: 0.5)
        XCTAssertEqual(mapped.x, 0.25, accuracy: 0.001)
        XCTAssertEqual(mapped.y, 0.5, accuracy: 0.001)
    }

    func testAnalyzeRequestUsesBackendMultipartContract() throws {
        let client = AnalysisAPIClient(baseURL: URL(string: "http://127.0.0.1:8000")!)
        let request = try client.makeAnalyzeRequest(
            swingID: "swing-1",
            durationMs: 7193,
            trimStartMs: 100,
            trimEndMs: 6200,
            videoData: Data("movie".utf8),
            filename: "swing-1.mp4"
        )

        XCTAssertEqual(request.url?.absoluteString, "http://127.0.0.1:8000/analyze-swing")
        XCTAssertEqual(request.httpMethod, "POST")
        XCTAssertTrue(request.value(forHTTPHeaderField: "Content-Type")?.contains("multipart/form-data") == true)

        let body = String(decoding: try XCTUnwrap(request.httpBody), as: UTF8.self)
        XCTAssertTrue(body.contains("name=\"swing_id\""))
        XCTAssertTrue(body.contains("swing-1"))
        XCTAssertTrue(body.contains("name=\"client_duration_ms\""))
        XCTAssertTrue(body.contains("7193"))
        XCTAssertTrue(body.contains("name=\"trim_start_ms\""))
        XCTAssertTrue(body.contains("name=\"trim_end_ms\""))
        XCTAssertTrue(body.contains("name=\"video\"; filename=\"swing-1.mp4\""))
    }

    func testAnalyzeRequestCanPreserveQuickTimeUploadMetadata() throws {
        let client = AnalysisAPIClient(baseURL: URL(string: "http://127.0.0.1:8000")!)
        let request = try client.makeAnalyzeRequest(
            swingID: "swing-1",
            durationMs: 7193,
            trimStartMs: nil,
            trimEndMs: nil,
            videoData: Data("movie".utf8),
            filename: "swing-1.mov",
            mimeType: "video/quicktime"
        )

        let body = String(decoding: try XCTUnwrap(request.httpBody), as: UTF8.self)
        XCTAssertTrue(body.contains("name=\"video\"; filename=\"swing-1.mov\""))
        XCTAssertTrue(body.contains("Content-Type: video/quicktime"))
    }

    func testAnalyzeRequestIncludesVideoEditFields() throws {
        let client = AnalysisAPIClient(baseURL: URL(string: "http://127.0.0.1:8000")!)
        let request = try client.makeAnalyzeRequest(
            swingID: "swing-1",
            durationMs: 7193,
            trimStartMs: 100,
            trimEndMs: 6200,
            videoEditState: VideoEditState(rotationDegrees: 90, isMirroredHorizontally: true),
            videoData: Data("movie".utf8),
            filename: "swing-1.mp4"
        )

        let body = String(decoding: try XCTUnwrap(request.httpBody), as: UTF8.self)
        XCTAssertTrue(body.contains("name=\"rotation_degrees\""))
        XCTAssertTrue(body.contains("90"))
        XCTAssertTrue(body.contains("name=\"mirror_horizontal\""))
        XCTAssertTrue(body.contains("true"))
    }

    func testAIFeedbackRequestUsesBackendMultipartContract() throws {
        let client = AnalysisAPIClient(baseURL: URL(string: "http://127.0.0.1:8000")!)
        let request = try client.makeAIFeedbackRequest(
            swingID: "swing-1",
            club: "7",
            analysisJSON: #"{"senseiScore":7.1}"#,
            durationMs: 7193,
            trimStartMs: 100,
            trimEndMs: 6200,
            videoEditState: VideoEditState(rotationDegrees: 270, isMirroredHorizontally: true),
            videoData: Data("movie".utf8),
            filename: "swing-1.mp4"
        )

        XCTAssertEqual(request.url?.absoluteString, "http://127.0.0.1:8000/ai-analysis")
        XCTAssertEqual(request.httpMethod, "POST")
        XCTAssertTrue(request.value(forHTTPHeaderField: "Content-Type")?.contains("multipart/form-data") == true)

        let body = String(decoding: try XCTUnwrap(request.httpBody), as: UTF8.self)
        XCTAssertTrue(body.contains("name=\"swing_id\""))
        XCTAssertTrue(body.contains("swing-1"))
        XCTAssertTrue(body.contains("name=\"club\""))
        XCTAssertTrue(body.contains("7"))
        XCTAssertTrue(body.contains("name=\"analysis_json\"; filename=\"swing-1-analysis.json\""))
        XCTAssertTrue(body.contains("Content-Type: application/json"))
        XCTAssertTrue(body.contains("senseiScore"))
        XCTAssertTrue(body.contains("name=\"client_duration_ms\""))
        XCTAssertTrue(body.contains("name=\"trim_start_ms\""))
        XCTAssertTrue(body.contains("name=\"trim_end_ms\""))
        XCTAssertTrue(body.contains("name=\"rotation_degrees\""))
        XCTAssertTrue(body.contains("270"))
        XCTAssertTrue(body.contains("name=\"mirror_horizontal\""))
        XCTAssertTrue(body.contains("true"))
        XCTAssertTrue(body.contains("name=\"video\"; filename=\"swing-1.mp4\""))
    }

    func testDecodesSwingAIFeedbackContract() throws {
        let feedback = try makeFixtureAIFeedback()

        XCTAssertEqual(feedback.provider, "gemini")
        XCTAssertEqual(feedback.club, "7")
        XCTAssertEqual(feedback.sourceAnalysisVersion, 21)
        XCTAssertEqual(feedback.coachScore, 7.8)
        XCTAssertEqual(feedback.priorityFixes.first?.cue, "Keep the sternum over the ball.")
        XCTAssertEqual(feedback.practicePlan.first?.reps, "3 sets of 8")
    }

    func testSwingRecordDecodesWithoutVideoEditState() throws {
        let json = """
        {
          "id": "legacy-swing",
          "videoURL": "file:///tmp/legacy.mov",
          "createdAt": "2023-11-14T22:13:20Z",
          "source": "uploaded",
          "orientation": "portrait",
          "status": "trimming"
        }
        """
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        let record = try decoder.decode(SwingRecord.self, from: Data(json.utf8))

        XCTAssertNil(record.videoEditState)
        XCTAssertNil(record.club)
        XCTAssertNil(record.aiAnalysis)
    }

    func testSwingRecordDecodesClubAndAIFeedback() throws {
        let json = """
        {
          "id": "ai-swing",
          "videoURL": "file:///tmp/ai.mov",
          "createdAt": "2023-11-14T22:13:20Z",
          "source": "uploaded",
          "orientation": "portrait",
          "club": "7",
          "aiAnalysis": \(fixtureAIFeedbackJSON),
          "status": "complete"
        }
        """
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        let record = try decoder.decode(SwingRecord.self, from: Data(json.utf8))

        XCTAssertEqual(record.club, "7")
        XCTAssertEqual(record.aiAnalysis?.headline, "Stable base with a late posture leak.")
    }

    func testSwingRepositoryPersistsSwingJSON() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let repository = SwingRepository(rootDirectory: root)
        let record = SwingRecord(
            id: "swing-json",
            videoURL: root.appendingPathComponent("videos/swing-json.mov"),
            createdAt: Date(timeIntervalSince1970: 1_700_000_000),
            source: .uploaded,
            orientation: .portrait,
            durationMs: 2_400,
            trimStartMs: 120,
            trimEndMs: 2_100,
            videoEditState: VideoEditState(rotationDegrees: 270, isMirroredHorizontally: true),
            senseiScore: 8.4,
            status: .complete
        )

        try repository.save([record])
        let loaded = try repository.load()

        XCTAssertEqual(loaded, [record])
        XCTAssertEqual(loaded.first?.videoEditState?.rotationDegrees, 270)
        XCTAssertEqual(loaded.first?.videoEditState?.isMirroredHorizontally, true)
    }

    func testSwingRepositoryCopiesVideoIntoStablePath() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("source.mov")
        try Data("video".utf8).write(to: source)

        let repository = SwingRepository(rootDirectory: root.appendingPathComponent("store"))
        let record = try repository.copyVideo(
            from: source,
            source: .recorded,
            durationMs: 3_100,
            orientation: .landscape,
            id: "fixed-id",
            createdAt: Date(timeIntervalSince1970: 1_700_000_100)
        )

        XCTAssertEqual(record.videoURL.lastPathComponent, "fixed-id.mov")
        XCTAssertTrue(FileManager.default.fileExists(atPath: record.videoURL.path))
        XCTAssertEqual(record.status, .trimming)
        XCTAssertEqual(record.trimStartMs, 0)
        XCTAssertEqual(record.trimEndMs, 3_100)
    }

    func testTrimWindowValidationClampsBoundsAndMinimumDuration() {
        let trim = TrimWindow.validated(startMs: 4_900, endMs: 5_100, durationMs: 5_000)

        XCTAssertEqual(trim.startMs, 4_200)
        XCTAssertEqual(trim.endMs, 5_000)
        XCTAssertTrue(TrimWindow.canContinue(startMs: 0, endMs: 900, durationMs: 5_000))
        XCTAssertFalse(TrimWindow.canContinue(startMs: 0, endMs: 400, durationMs: 5_000))
    }

    func testNormalizesRelativeDebugURLsAgainstBackendBaseURL() throws {
        let analysis = try makeFixtureAnalysis()
        let normalized = analysis.normalizingDebugURLs(baseURL: URL(string: "http://127.0.0.1:8000")!)

        XCTAssertEqual(normalized.debug?.contactSheetUrl, "http://127.0.0.1:8000/debug/fixture_phases_v20.jpg")
        XCTAssertEqual(normalized.debug?.impactFilmstripUrl, "http://127.0.0.1:8000/debug/fixture_impact_v20.jpg")
    }

    private func makeClubPathFrames(points: [SwingPoint?]) -> [SwingAnalysisFrame] {
        points.enumerated().map { index, point in
            SwingAnalysisFrame(
                t: index * 100,
                keypoints: makeClubPathKeypoints(),
                clubHead: point
            )
        }
    }

    private func makeClubPathKeypoints() -> [SwingKeypoint] {
        [
            SwingKeypoint(name: "nose", x: 0.50, y: 0.24, score: 0.95),
            SwingKeypoint(name: "left_shoulder", x: 0.43, y: 0.38, score: 0.95),
            SwingKeypoint(name: "right_shoulder", x: 0.57, y: 0.38, score: 0.95),
            SwingKeypoint(name: "left_elbow", x: 0.46, y: 0.59, score: 0.95),
            SwingKeypoint(name: "right_elbow", x: 0.54, y: 0.59, score: 0.95),
            SwingKeypoint(name: "left_wrist", x: 0.485, y: 0.70, score: 0.95),
            SwingKeypoint(name: "right_wrist", x: 0.515, y: 0.70, score: 0.95),
            SwingKeypoint(name: "left_hip", x: 0.44, y: 0.58, score: 0.95),
            SwingKeypoint(name: "right_hip", x: 0.56, y: 0.58, score: 0.95),
            SwingKeypoint(name: "left_knee", x: 0.45, y: 0.76, score: 0.95),
            SwingKeypoint(name: "right_knee", x: 0.55, y: 0.76, score: 0.95),
            SwingKeypoint(name: "left_ankle", x: 0.45, y: 0.94, score: 0.95),
            SwingKeypoint(name: "right_ankle", x: 0.55, y: 0.94, score: 0.95),
            SwingKeypoint(name: "left_heel", x: 0.435, y: 0.95, score: 0.95),
            SwingKeypoint(name: "right_heel", x: 0.565, y: 0.95, score: 0.95),
            SwingKeypoint(name: "left_foot_index", x: 0.415, y: 0.955, score: 0.95),
            SwingKeypoint(name: "right_foot_index", x: 0.585, y: 0.955, score: 0.95),
        ]
    }

    private func makeFixtureAnalysis(cameraAngle: String = "down_the_line") throws -> SwingAnalysis {
        let json = """
        {
          "id": "fixture",
          "videoUri": "file:///fixture.mp4",
          "fps": 6,
          "durationMs": 2400,
          "analysisVersion": 20,
          "debug": {
            "contactSheetUrl": "/debug/fixture_phases_v20.jpg",
            "impactFilmstripUrl": "/debug/fixture_impact_v20.jpg",
            "impactWindowTimesMs": [
              { "analysis": 1000, "frame": 2, "offset": 0, "selected": true, "source": 2009 }
            ],
            "positionTimesMs": {
              "impact": { "analysis": 1000, "source": 2009 }
            },
            "analysisQuality": {
              "status": "ok",
              "cameraAngle": "\(cameraAngle)",
              "captureConfidence": 0.91,
              "scoreConfidence": 0.9,
              "poseCoverage": 0.95,
              "bodyVisibility": 0.88,
              "golferScale": 0.52,
              "framing": 0.96,
              "warnings": []
            },
            "phaseConfidence": {
              "impact": 0.92
            }
          },
          "videoWidth": 1080,
          "videoHeight": 1920,
          "trimStartMs": 0,
          "trimEndMs": 2400,
          "frames": [
            { "t": 0, "keypoints": [] },
            { "t": 600, "keypoints": [] },
            { "t": 1000, "keypoints": [
              { "name": "left_wrist", "x": 0.4, "y": 0.5, "score": 0.9 },
              { "name": "right_wrist", "x": 0.5, "y": 0.5, "score": 0.9 }
            ], "clubHead": { "x": 0.62, "y": 0.78 } },
            { "t": 1500, "keypoints": [] }
          ],
          "positions": {
            "address": 0,
            "takeaway": 1,
            "top": 1,
            "impact": 2,
            "finish": 3
          },
          "metrics": {
            "headDisplacement": 0.01,
            "shoulderTurn": 22,
            "hipSway": 0.02,
            "postureLoss": 2,
            "tempo": 1.2
          },
          "feedback": [
            {
              "position": "impact",
              "label": "Posture maintained",
              "status": "good",
              "detail": "Keep it."
            }
          ],
          "swingThought": {
            "position": "impact",
            "label": "Posture through impact",
            "status": "warning",
            "thought": "Keep your chest over the ball.",
            "reason": "Your spine angle is changing before impact.",
            "metricKey": "postureLoss"
          },
          "analysisQuality": {
            "status": "ok",
            "cameraAngle": "\(cameraAngle)",
            "captureConfidence": 0.91,
            "scoreConfidence": 0.9,
            "poseCoverage": 0.95,
            "bodyVisibility": 0.88,
            "golferScale": 0.52,
            "framing": 0.96,
            "warnings": []
          },
          "phaseConfidence": {
            "address": 0.89,
            "takeaway": 0.86,
            "top": 0.88,
            "impact": 0.92,
            "finish": 0.84
          },
          "rawSenseiScore": 8.4,
          "senseiScore": 8.4
        }
        """
        return try JSONDecoder().decode(SwingAnalysis.self, from: Data(json.utf8))
    }

    private var fixtureAIFeedbackJSON: String {
        """
        {
          "id": "ai-fixture",
          "provider": "gemini",
          "model": "gemini-2.5-pro",
          "generatedAt": "2026-05-13T12:00:00Z",
          "club": "7",
          "sourceAnalysisVersion": 21,
          "coachScore": 7.8,
          "headline": "Stable base with a late posture leak.",
          "summary": "Your address is organized, but your chest rises through impact. Keep your posture quieter so the low point stays predictable.",
          "confidence": "medium",
          "strengths": [
            "Balanced setup",
            "Good tempo match between backswing and downswing"
          ],
          "priorityFixes": [
            {
              "title": "Posture through impact",
              "phase": "impact",
              "observation": "The upper body rises before contact.",
              "whyItMatters": "It moves the club bottom and makes contact inconsistent.",
              "cue": "Keep the sternum over the ball."
            }
          ],
          "practicePlan": [
            {
              "title": "Slow impact rehearsals",
              "reps": "3 sets of 8",
              "focus": "Pause at impact with chest tilt intact."
            }
          ],
          "limitations": [
            "Ball flight is not visible."
          ]
        }
        """
    }

    private func makeFixtureAIFeedback() throws -> SwingAIFeedback {
        try JSONDecoder().decode(SwingAIFeedback.self, from: Data(fixtureAIFeedbackJSON.utf8))
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("SwingSenseiCoreTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}
