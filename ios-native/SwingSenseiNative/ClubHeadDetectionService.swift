import Foundation

final class ClubHeadDetectionService {
    private let detector: ClubHeadDetector

    init(detector: ClubHeadDetector) {
        self.detector = detector
    }

    func populateClubHead(
        into analysis: SwingAnalysis,
        videoURL: URL
    ) async throws -> SwingAnalysis {
        let sampler = VideoFrameSampler(url: videoURL)

        var newFrames: [SwingAnalysisFrame] = []
        newFrames.reserveCapacity(analysis.frames.count)

        var detectedCount = 0
        var extractionErrors = 0
        var detectionErrors = 0
        var sampledPoints: [(Int, Double, Double, Float)] = []

        for frame in analysis.frames {
            var clubHead: SwingPoint? = nil
            do {
                let image = try await sampler.image(atMs: frame.t)
                do {
                    if let detection = try detector.detect(in: image) {
                        clubHead = SwingPoint(
                            x: Double(detection.center.x),
                            y: Double(detection.center.y)
                        )
                        detectedCount += 1
                        if sampledPoints.count < 5 {
                            sampledPoints.append((frame.t, Double(detection.center.x), Double(detection.center.y), detection.confidence))
                        }
                    }
                } catch {
                    detectionErrors += 1
                }
            } catch {
                extractionErrors += 1
            }
            newFrames.append(
                SwingAnalysisFrame(t: frame.t, keypoints: frame.keypoints, clubHead: clubHead)
            )
        }

        newFrames = cleanedAndBridged(newFrames)
        let postCleanCount = newFrames.filter { $0.clubHead != nil }.count
        print("[ClubHeadDetectionService] clean+bridge: clubHead frames \(detectedCount) -> \(postCleanCount)")

        print("[ClubHeadDetectionService] frames=\(analysis.frames.count) detected=\(detectedCount) extractionErrors=\(extractionErrors) detectionErrors=\(detectionErrors)")
        for (t, x, y, conf) in sampledPoints {
            print("[ClubHeadDetectionService] sample t=\(t)ms x=\(String(format: "%.3f", x)) y=\(String(format: "%.3f", y)) conf=\(String(format: "%.2f", conf))")
        }

        // Segment statistics: how do detections cluster into consecutive-frame runs?
        struct Run { var startT: Int; var endT: Int; var points: [(Double, Double)] }
        var runs: [Run] = []
        var current: Run? = nil
        for f in newFrames {
            if let ch = f.clubHead {
                if current == nil {
                    current = Run(startT: f.t, endT: f.t, points: [(ch.x, ch.y)])
                } else {
                    current!.endT = f.t
                    current!.points.append((ch.x, ch.y))
                }
            } else {
                if let c = current { runs.append(c) }
                current = nil
            }
        }
        if let c = current { runs.append(c) }
        let longest = runs.max(by: { $0.points.count < $1.points.count })?.points.count ?? 0
        let runsOf3Plus = runs.filter { $0.points.count >= 3 }.count
        print("[ClubHeadDetectionService] segments: total=\(runs.count) longest=\(longest) runsOf3+=\(runsOf3Plus)")
        for (i, r) in runs.enumerated() where r.points.count >= 3 {
            let xs = r.points.map(\.0)
            let ys = r.points.map(\.1)
            let dx = (xs.max() ?? 0) - (xs.min() ?? 0)
            let dy = (ys.max() ?? 0) - (ys.min() ?? 0)
            let displacement = (dx*dx + dy*dy).squareRoot()
            print("[ClubHeadDetectionService] run #\(i): \(r.points.count) samples, t=\(r.startT)..\(r.endT)ms, displacement=\(String(format: "%.3f", displacement)) (need >=0.035)")
        }

        // What survives the actual stableSegments pipeline (incl. isPlausible body-box check)?
        let stableSegments = ClubPathStabilizer.stableSegments(from: newFrames)
        print("[ClubHeadDetectionService] ClubPathStabilizer.stableSegments returned \(stableSegments.count) segment(s)")
        for (i, seg) in stableSegments.enumerated() {
            print("[ClubHeadDetectionService] stable #\(i): \(seg.samples.count) samples")
        }

        // Pinpoint which filter fragments the swing arc.
        let report = ClubPathStabilizer.plausibilityReport(for: newFrames)
        let detected = report.filter { $0.hasClubHead }
        let implausible = detected.filter { !$0.plausible }
        let bigSteps = detected.filter {
            guard let s = $0.stepFromPrevDetection, let l = $0.stepLimit else { return false }
            return s > l
        }
        print("[ClubHeadDetectionService] filter breakdown: detected=\(detected.count) rejectedByPlausibility=\(implausible.count) brokenByStepLimit=\(bigSteps.count)")
        for r in implausible.prefix(8) {
            print("[ClubHeadDetectionService]   implausible t=\(r.t)ms")
        }
        for r in bigSteps.prefix(8) {
            print("[ClubHeadDetectionService]   stepBreak t=\(r.t)ms step=\(String(format: "%.3f", r.stepFromPrevDetection ?? 0)) limit=\(String(format: "%.3f", r.stepLimit ?? 0))")
        }

        // Per-segment scores vs primary-segment thresholds.
        let scores = ClubPathStabilizer.scoreReport(from: newFrames, positions: analysis.positions)
        print("[ClubHeadDetectionService] segment scores (full: score>=0.58 phase>=0.25 throughImpact>=0.35 | partial: score>=0.34 throughImpact>=0.55):")
        for (i, s) in scores.enumerated() {
            print("[ClubHeadDetectionService]   seg #\(i): samples=\(s.sampleCount) disp=\(String(format: "%.3f", s.displacement)) score=\(String(format: "%.3f", s.score)) phase=\(String(format: "%.3f", s.phaseScore)) throughImpact=\(String(format: "%.3f", s.throughImpactScore))")
        }

        let cleanedPath = newFrames.compactMap(\.clubHead)
        let isDownTheLine = analysis.analysisQuality?.cameraAngle == "down_the_line"
        let swingPathDiagnosis = SwingPathDiagnoser.diagnose(
            path: cleanedPath,
            isDownTheLine: isDownTheLine
        )
        if let swingPathDiagnosis {
            print("[ClubHeadDetectionService] swing path diagnosis: \(swingPathDiagnosis.direction.rawValue) severity=\(String(format: "%.3f", swingPathDiagnosis.severity))")
        } else {
            print("[ClubHeadDetectionService] swing path diagnosis: nil (downTheLine=\(isDownTheLine) pathPoints=\(cleanedPath.count))")
        }

        return SwingAnalysis(
            id: analysis.id,
            videoUri: analysis.videoUri,
            fps: analysis.fps,
            durationMs: analysis.durationMs,
            analysisVersion: analysis.analysisVersion,
            debug: analysis.debug,
            videoWidth: analysis.videoWidth,
            videoHeight: analysis.videoHeight,
            trimStartMs: analysis.trimStartMs,
            trimEndMs: analysis.trimEndMs,
            frames: newFrames,
            positions: analysis.positions,
            metrics: analysis.metrics,
            feedback: analysis.feedback,
            swingThought: analysis.swingThought,
            analysisQuality: analysis.analysisQuality,
            phaseConfidence: analysis.phaseConfidence,
            shotEstimate: analysis.shotEstimate,
            rawSenseiScore: analysis.rawSenseiScore,
            senseiScore: analysis.senseiScore,
            swingPathDiagnosis: swingPathDiagnosis
        )
    }

    /// Post-process raw detections: drop spike outliers, then bridge small gaps
    /// so the swing fragments merge into one continuous path.
    private func cleanedAndBridged(_ frames: [SwingAnalysisFrame]) -> [SwingAnalysisFrame] {
        var points: [SwingPoint?] = frames.map(\.clubHead)
        func detectedIndices() -> [Int] { points.indices.filter { points[$0] != nil } }

        // Pass 1: drop interior spikes — a detection far off the line between its
        // neighbouring detections (e.g. a single frame that latched onto a shadow).
        let outlierDeviation = 0.18
        let idx1 = detectedIndices()
        if idx1.count >= 3 {
            for k in 1..<(idx1.count - 1) {
                let prev = idx1[k - 1], cur = idx1[k], next = idx1[k + 1]
                guard let p = points[prev], let c = points[cur], let n = points[next] else { continue }
                let frac = Double(cur - prev) / Double(next - prev)
                let ex = p.x + (n.x - p.x) * frac
                let ey = p.y + (n.y - p.y) * frac
                if (hypot(c.x - ex, c.y - ey)) > outlierDeviation {
                    points[cur] = nil
                }
            }
        }

        // Pass 2: trim trailing detections that sharply reverse direction — the
        // follow-through moves smoothly; a jump reversing into the shadow does not.
        var idx2 = detectedIndices()
        while idx2.count >= 3 {
            let a = points[idx2[idx2.count - 3]]!
            let b = points[idx2[idx2.count - 2]]!
            let l = points[idx2[idx2.count - 1]]!
            let v1x = b.x - a.x, v1y = b.y - a.y
            let v2x = l.x - b.x, v2y = l.y - b.y
            let m1 = hypot(v1x, v1y), m2 = hypot(v2x, v2y)
            let cosAngle = (m1 > 0.0001 && m2 > 0.0001)
                ? (v1x * v2x + v1y * v2y) / (m1 * m2) : 1.0
            if m2 > 0.05 && cosAngle < -0.5 {
                points[idx2[idx2.count - 1]] = nil
                idx2.removeLast()
            } else {
                break
            }
        }

        // Pass 3: bridge small detection gaps with linear interpolation.
        let maxGapFrames = 7
        let idx3 = detectedIndices()
        if idx3.count >= 2 {
            for k in 0..<(idx3.count - 1) {
                let lo = idx3[k], hi = idx3[k + 1]
                let gap = hi - lo - 1
                guard gap >= 1, gap <= maxGapFrames,
                      let p = points[lo], let n = points[hi] else { continue }
                for j in (lo + 1)..<hi {
                    let frac = Double(j - lo) / Double(hi - lo)
                    points[j] = SwingPoint(x: p.x + (n.x - p.x) * frac,
                                           y: p.y + (n.y - p.y) * frac)
                }
            }
        }

        // Pass 4 (temporary): trim the path at impact, removing the follow-through.
        // Top of swing = highest point on screen (min y). Impact = the FIRST y-peak
        // after the top (not the global max — the golfer lowers the club to a rest
        // afterwards, which sits even lower on screen). Confirmed by the follow-
        // through rising clear of that peak.
        let idx4 = detectedIndices()
        if idx4.count >= 5,
           let topPos = idx4.min(by: { points[$0]!.y < points[$1]!.y }) {
            var maxY = points[topPos]!.y
            var impactPos = topPos
            var foundImpact = false
            for i in idx4 where i > topPos {
                let y = points[i]!.y
                if y >= maxY {
                    maxY = y
                    impactPos = i
                } else if maxY - y > 0.10 {
                    foundImpact = true
                    break
                }
            }
            if foundImpact {
                for j in idx4 where j > impactPos { points[j] = nil }
            }
        }

        return zip(frames, points).map { frame, point in
            SwingAnalysisFrame(t: frame.t, keypoints: frame.keypoints, clubHead: point)
        }
    }
}
