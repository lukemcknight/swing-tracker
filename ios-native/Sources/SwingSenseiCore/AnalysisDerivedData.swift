import CoreGraphics
import Foundation

public struct PhaseMarker: Equatable, Identifiable {
    public var id: SwingPosition { position }

    public let position: SwingPosition
    public let frameIndex: Int
    public let timeMs: Int
    public let progress: Double

    public var label: String { position.label }
}

public struct ClubPathSample: Equatable, Identifiable {
    public var id: Int { frameIndex }

    public let frameIndex: Int
    public let frame: SwingAnalysisFrame
    public let point: SwingPoint

    public init(frameIndex: Int, frame: SwingAnalysisFrame, point: SwingPoint) {
        self.frameIndex = frameIndex
        self.frame = frame
        self.point = point
    }
}

public struct ClubPathSegment: Equatable, Identifiable {
    public var id: Int { samples.first?.frameIndex ?? 0 }

    public let samples: [ClubPathSample]

    public init(samples: [ClubPathSample]) {
        self.samples = samples
    }

    public var points: [SwingPoint] {
        samples.map(\.point)
    }

    public var frames: [SwingAnalysisFrame] {
        samples.map(\.frame)
    }
}

public enum ClubPathStabilizer {
    private struct ScoredSegment {
        let segment: ClubPathSegment
        let score: Double
        let phaseScore: Double
        let throughImpactScore: Double
    }

    private static let drawableScore = 0.2
    private static let reliableScore = 0.35
    private static let minimumSegmentPoints = 3
    private static let maximumFrameGap = 1
    private static let minimumPathDisplacement = 0.035
    private static let minimumPrimaryScore = 0.58
    private static let minimumPrimaryPhaseScore = 0.25
    private static let minimumPrimaryThroughImpactRatio = 0.35
    private static let ambiguousPrimaryScoreRatio = 0.86
    private static let ambiguousPhaseScoreGap = 0.25
    private static let maximumStepMinimum = 0.16
    private static let maximumStepMaximum = 0.34
    private static let maximumStepBodyHeightRatio = 0.55
    private static let minimumGripDistanceBodyHeightRatio = 0.045
    private static let maximumGripDistanceBodyHeightRatio = 1.08
    private static let bodyKeypointNames: Set<String> = [
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ]

    public static func stableSegments(
        from frames: [SwingAnalysisFrame],
        startingFrameIndex: Int = 0
    ) -> [ClubPathSegment] {
        var segments: [ClubPathSegment] = []
        var currentSamples: [ClubPathSample] = []

        func flushCurrentSamples() {
            guard isRenderableSegment(currentSamples) else {
                currentSamples.removeAll()
                return
            }

            segments.append(ClubPathSegment(samples: currentSamples))
            currentSamples.removeAll()
        }

        for (offset, frame) in frames.enumerated() {
            guard let point = frame.clubHead, isPlausible(point: point, in: frame) else {
                flushCurrentSamples()
                continue
            }

            let sample = ClubPathSample(
                frameIndex: startingFrameIndex + offset,
                frame: frame,
                point: point
            )
            guard let previous = currentSamples.last else {
                currentSamples = [sample]
                continue
            }

            let frameGap = sample.frameIndex - previous.frameIndex
            let stepLimit = clubHeadStepLimit(previous.frame, sample.frame)
            if frameGap <= maximumFrameGap && normalizedDistance(previous.point, sample.point) <= stepLimit {
                currentSamples.append(sample)
            } else {
                flushCurrentSamples()
                currentSamples = [sample]
            }
        }

        flushCurrentSamples()
        return segments
    }

    public static func primarySegment(
        from frames: [SwingAnalysisFrame],
        startingFrameIndex: Int = 0,
        positions: [String: Int] = [:]
    ) -> ClubPathSegment? {
        let requiredScore = positions.isEmpty ? 0.34 : minimumPrimaryScore
        let scoredSegments = stableSegments(from: frames, startingFrameIndex: startingFrameIndex)
            .map { segment in
                score(segment: segment, positions: positions)
            }
            .filter { $0.score >= requiredScore }
            .filter { positions.isEmpty || $0.phaseScore >= minimumPrimaryPhaseScore }
            .filter { positions.isEmpty || $0.throughImpactScore >= minimumPrimaryThroughImpactRatio }
            .sorted {
                if $0.score == $1.score {
                    return $0.segment.samples.count > $1.segment.samples.count
                }
                return $0.score > $1.score
            }

        guard let best = scoredSegments.first else { return nil }
        if
            scoredSegments.count > 1,
            scoredSegments[1].score >= best.score * ambiguousPrimaryScoreRatio,
            best.phaseScore - scoredSegments[1].phaseScore < ambiguousPhaseScoreGap
        {
            return nil
        }

        return best.segment
    }

    private static func isRenderableSegment(_ samples: [ClubPathSample]) -> Bool {
        samples.count >= minimumSegmentPoints && pathDisplacement(samples.map(\.point)) >= minimumPathDisplacement
    }

    private static func score(segment: ClubPathSegment, positions: [String: Int]) -> ScoredSegment {
        let samples = segment.samples
        let lengthScore = min(1.0, Double(samples.count) / 12.0)
        let displacementScore = min(1.0, pathDisplacement(segment.points) / 0.22)
        let continuityScore = continuityScore(samples)
        let phaseScores = phaseAlignmentScores(samples: samples, positions: positions)
        let phaseScore = phaseScores.phase
        let score = lengthScore * 0.34 + displacementScore * 0.26 + continuityScore * 0.18 + phaseScore * 0.42
        return ScoredSegment(
            segment: segment,
            score: score,
            phaseScore: phaseScore,
            throughImpactScore: phaseScores.throughImpact
        )
    }

    private static func continuityScore(_ samples: [ClubPathSample]) -> Double {
        guard samples.count >= 2 else { return 0 }
        let ratios = zip(samples, samples.dropFirst()).map { previous, current in
            let limit = clubHeadStepLimit(previous.frame, current.frame)
            return min(1.0, normalizedDistance(previous.point, current.point) / max(0.001, limit))
        }
        let averageRatio = ratios.reduce(0, +) / Double(max(1, ratios.count))
        return max(0, 1.0 - averageRatio)
    }

    private static func phaseAlignmentScore(samples: [ClubPathSample], positions: [String: Int]) -> Double {
        phaseAlignmentScores(samples: samples, positions: positions).phase
    }

    private static func phaseAlignmentScores(
        samples: [ClubPathSample],
        positions: [String: Int]
    ) -> (phase: Double, throughImpact: Double) {
        guard !samples.isEmpty else { return (0, 0) }
        let frameIndices = samples.map(\.frameIndex)
        guard
            let address = positions["address"],
            let finish = positions["finish"],
            finish >= address
        else {
            return (0, 0)
        }

        let impact = positions["impact"] ?? finish
        let top = positions["top"] ?? impact
        let inSwing = overlapRatio(frameIndices, lowerBound: address, upperBound: finish)
        let throughImpact = anchorRatio(frameIndices, lowerBound: address, upperBound: max(address, impact))
        let backswingToImpact = anchorRatio(frameIndices, lowerBound: address, upperBound: max(address, top))
        let outsidePenalty = 1.0 - inSwing
        let phase = max(
            0,
            min(
                1,
                inSwing * 0.50
                    + throughImpact * 0.30
                    + backswingToImpact * 0.15
                    - outsidePenalty * 0.45
            )
        )
        return (phase, throughImpact)
    }

    private static func overlapRatio(_ frameIndices: [Int], lowerBound: Int, upperBound: Int) -> Double {
        guard !frameIndices.isEmpty, upperBound >= lowerBound else { return 0 }
        let count = frameIndices.filter { lowerBound <= $0 && $0 <= upperBound }.count
        return Double(count) / Double(frameIndices.count)
    }

    private static func anchorRatio(_ frameIndices: [Int], lowerBound: Int, upperBound: Int) -> Double {
        guard !frameIndices.isEmpty, upperBound >= lowerBound else { return 0 }
        let count = frameIndices.filter { lowerBound <= $0 && $0 <= upperBound }.count
        let requiredAnchors = max(2, min(minimumSegmentPoints, upperBound - lowerBound + 1))
        return min(1, Double(count) / Double(requiredAnchors))
    }

    private static func isPlausible(point: SwingPoint, in frame: SwingAnalysisFrame) -> Bool {
        guard point.x.isFinite, point.y.isFinite, (0...1).contains(point.x), (0...1).contains(point.y) else {
            return false
        }
        guard let grip = gripPoint(in: frame), let box = bodyBox(in: frame) else {
            return true
        }

        let bodyHeight = clubReferenceBodyHeight(in: frame)
        let gripDistance = normalizedDistance(point, grip)
        let minimumGripDistance = max(0.018, bodyHeight * minimumGripDistanceBodyHeightRatio)
        let maximumGripDistance = min(0.86, max(0.24, bodyHeight * maximumGripDistanceBodyHeightRatio))
        guard gripDistance >= minimumGripDistance, gripDistance <= maximumGripDistance else {
            return false
        }

        return !contains(point: point, in: box, inset: 0.02)
    }

    private static func clubHeadStepLimit(_ firstFrame: SwingAnalysisFrame, _ secondFrame: SwingAnalysisFrame) -> Double {
        let bodyHeight = max(clubReferenceBodyHeight(in: firstFrame), clubReferenceBodyHeight(in: secondFrame))
        return max(maximumStepMinimum, min(maximumStepMaximum, bodyHeight * maximumStepBodyHeightRatio))
    }

    private static func clubReferenceBodyHeight(in frame: SwingAnalysisFrame) -> Double {
        guard let box = bodyBox(in: frame) else { return 0.45 }
        return max(0.24, min(0.72, box.maxY - box.minY))
    }

    private static func pathDisplacement(_ points: [SwingPoint]) -> Double {
        guard points.count >= 2 else { return 0 }
        let minX = points.map(\.x).min() ?? 0
        let maxX = points.map(\.x).max() ?? 0
        let minY = points.map(\.y).min() ?? 0
        let maxY = points.map(\.y).max() ?? 0
        return hypot(maxX - minX, maxY - minY)
    }

    private static func normalizedDistance(_ first: SwingPoint, _ second: SwingPoint) -> Double {
        hypot(first.x - second.x, first.y - second.y)
    }

    private static func gripPoint(in frame: SwingAnalysisFrame) -> SwingPoint? {
        let keypoints = Dictionary(uniqueKeysWithValues: frame.keypoints.map { ($0.name, $0) })
        let left = keypoints["left_wrist"]
        let right = keypoints["right_wrist"]
        let leftScore = left?.score ?? 0
        let rightScore = right?.score ?? 0

        if leftScore >= reliableScore, rightScore >= reliableScore {
            return weightedMidpoint(left, right)
        }
        if leftScore >= reliableScore, let left {
            return SwingPoint(x: left.x, y: left.y)
        }
        if rightScore >= reliableScore, let right {
            return SwingPoint(x: right.x, y: right.y)
        }
        if leftScore >= drawableScore, rightScore >= drawableScore {
            return weightedMidpoint(left, right)
        }
        if leftScore >= drawableScore, let left {
            return SwingPoint(x: left.x, y: left.y)
        }
        if rightScore >= drawableScore, let right {
            return SwingPoint(x: right.x, y: right.y)
        }

        return nil
    }

    private static func weightedMidpoint(_ first: SwingKeypoint?, _ second: SwingKeypoint?) -> SwingPoint? {
        guard let first, let second else { return nil }
        let firstWeight = max(0.05, first.score)
        let secondWeight = max(0.05, second.score)
        let totalWeight = firstWeight + secondWeight
        guard totalWeight > 0 else { return nil }

        return SwingPoint(
            x: (first.x * firstWeight + second.x * secondWeight) / totalWeight,
            y: (first.y * firstWeight + second.y * secondWeight) / totalWeight
        )
    }

    private static func bodyBox(in frame: SwingAnalysisFrame) -> (minX: Double, maxX: Double, minY: Double, maxY: Double)? {
        let points = frame.keypoints.filter { bodyKeypointNames.contains($0.name) && $0.score >= drawableScore }
        guard points.count >= 4 else { return nil }

        return (
            minX: points.map(\.x).min() ?? 0,
            maxX: points.map(\.x).max() ?? 0,
            minY: points.map(\.y).min() ?? 0,
            maxY: points.map(\.y).max() ?? 0
        )
    }

    private static func contains(
        point: SwingPoint,
        in box: (minX: Double, maxX: Double, minY: Double, maxY: Double),
        inset: Double
    ) -> Bool {
        let minX = box.minX + inset
        let maxX = box.maxX - inset
        let minY = box.minY + inset
        let maxY = box.maxY - inset
        guard minX < maxX, minY < maxY else { return false }

        return minX <= point.x && point.x <= maxX && minY <= point.y && point.y <= maxY
    }
}

public extension SwingAnalysis {
    func marker(for position: SwingPosition) -> PhaseMarker? {
        guard let index = frameIndex(for: position) else { return nil }
        let timeMs = frames[index].t
        return PhaseMarker(
            position: position,
            frameIndex: index,
            timeMs: timeMs,
            progress: Double(timeMs) / Double(max(1, durationMs))
        )
    }

    func phaseMarkers() -> [PhaseMarker] {
        SwingPosition.allCases.compactMap { marker(for: $0) }
    }

    func nearestFrameIndex(to timeMs: Double) -> Int? {
        guard !frames.isEmpty else { return nil }

        var low = 0
        var high = frames.count - 1
        while low < high {
            let mid = (low + high) / 2
            if Double(frames[mid].t) < timeMs {
                low = mid + 1
            } else {
                high = mid
            }
        }

        if low == 0 { return 0 }
        let previous = low - 1
        let previousDistance = abs(Double(frames[previous].t) - timeMs)
        let currentDistance = abs(Double(frames[low].t) - timeMs)
        return previousDistance <= currentDistance ? previous : low
    }

    func closestPosition(to timeMs: Double) -> SwingPosition {
        phaseMarkers().min { lhs, rhs in
            abs(Double(lhs.timeMs) - timeMs) < abs(Double(rhs.timeMs) - timeMs)
        }?.position ?? .takeaway
    }

    func stableClubPathSegments(in range: ClosedRange<Int>? = nil) -> [ClubPathSegment] {
        guard !frames.isEmpty else { return [] }

        let lowerBound = max(0, range?.lowerBound ?? 0)
        let upperBound = min(frames.count - 1, range?.upperBound ?? frames.count - 1)
        guard lowerBound <= upperBound else { return [] }

        return ClubPathStabilizer.stableSegments(
            from: Array(frames[lowerBound...upperBound]),
            startingFrameIndex: lowerBound
        )
    }

    func primaryClubPathSegment(in range: ClosedRange<Int>? = nil) -> ClubPathSegment? {
        guard !frames.isEmpty else { return nil }

        let lowerBound = max(0, range?.lowerBound ?? 0)
        let upperBound = min(frames.count - 1, range?.upperBound ?? frames.count - 1)
        guard lowerBound <= upperBound else { return nil }

        return ClubPathStabilizer.primarySegment(
            from: Array(frames[lowerBound...upperBound]),
            startingFrameIndex: lowerBound,
            positions: positions
        )
    }
}

public enum VideoGeometry {
    public static func aspectFitRect(videoSize: CGSize, containerSize: CGSize) -> CGRect {
        guard videoSize.width > 0, videoSize.height > 0, containerSize.width > 0, containerSize.height > 0 else {
            return .zero
        }

        let scale = min(containerSize.width / videoSize.width, containerSize.height / videoSize.height)
        let width = videoSize.width * scale
        let height = videoSize.height * scale
        return CGRect(
            x: (containerSize.width - width) / 2,
            y: (containerSize.height - height) / 2,
            width: width,
            height: height
        )
    }

    public static func mapNormalizedPoint(x: Double, y: Double, into videoRect: CGRect) -> CGPoint {
        CGPoint(
            x: videoRect.minX + CGFloat(x) * videoRect.width,
            y: videoRect.minY + CGFloat(y) * videoRect.height
        )
    }

    public static func presentationSize(naturalSize: CGSize, preferredTransform: CGAffineTransform) -> CGSize {
        let bounds = transformedBounds(size: naturalSize, transform: preferredTransform)
        return CGSize(width: abs(bounds.width), height: abs(bounds.height))
    }

    static func transformedBounds(size: CGSize, transform: CGAffineTransform) -> CGRect {
        let corners = [
            CGPoint(x: 0, y: 0),
            CGPoint(x: size.width, y: 0),
            CGPoint(x: 0, y: size.height),
            CGPoint(x: size.width, y: size.height),
        ].map { $0.applying(transform) }

        let minX = corners.map(\.x).min() ?? 0
        let maxX = corners.map(\.x).max() ?? 0
        let minY = corners.map(\.y).min() ?? 0
        let maxY = corners.map(\.y).max() ?? 0
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }
}

public struct VideoCoordinateSpace {
    public let sourceSize: CGSize
    public let displaySize: CGSize

    private let sourceToDisplayTransform: CGAffineTransform?
    private let transformedBounds: CGRect

    public static func identity(size: CGSize) -> VideoCoordinateSpace {
        VideoCoordinateSpace(sourceSize: size)
    }

    public static func displayingAsset(
        analysisSize: CGSize,
        naturalSize: CGSize,
        preferredTransform: CGAffineTransform
    ) -> VideoCoordinateSpace {
        let safeAnalysisSize = safeSize(analysisSize)
        let safeNaturalSize = safeSize(naturalSize)
        let presentationSize = safeSize(
            VideoGeometry.presentationSize(naturalSize: safeNaturalSize, preferredTransform: preferredTransform)
        )

        if aspectRatiosMatch(safeAnalysisSize, presentationSize) {
            return VideoCoordinateSpace(sourceSize: safeAnalysisSize, displaySize: presentationSize)
        }

        if aspectRatiosMatch(safeAnalysisSize, safeNaturalSize),
           !aspectRatiosMatch(safeNaturalSize, presentationSize),
           !preferredTransform.isIdentity {
            return VideoCoordinateSpace(
                sourceSize: safeAnalysisSize,
                displaySize: presentationSize,
                sourceToDisplayTransform: preferredTransform
            )
        }

        return VideoCoordinateSpace(sourceSize: safeAnalysisSize)
    }

    public init(
        sourceSize: CGSize,
        displaySize: CGSize? = nil,
        sourceToDisplayTransform: CGAffineTransform? = nil
    ) {
        let safeSourceSize = Self.safeSize(sourceSize)
        self.sourceSize = safeSourceSize
        self.displaySize = Self.safeSize(displaySize ?? safeSourceSize)

        if let sourceToDisplayTransform, !sourceToDisplayTransform.isIdentity {
            self.sourceToDisplayTransform = sourceToDisplayTransform
            self.transformedBounds = VideoGeometry.transformedBounds(
                size: safeSourceSize,
                transform: sourceToDisplayTransform
            )
        } else {
            self.sourceToDisplayTransform = nil
            self.transformedBounds = CGRect(origin: .zero, size: safeSourceSize)
        }
    }

    public func mapNormalizedPoint(x: Double, y: Double) -> SwingPoint {
        guard
            let sourceToDisplayTransform,
            transformedBounds.width > 0,
            transformedBounds.height > 0
        else {
            return SwingPoint(x: x, y: y)
        }

        let sourcePoint = CGPoint(
            x: CGFloat(x) * sourceSize.width,
            y: CGFloat(y) * sourceSize.height
        ).applying(sourceToDisplayTransform)
        let normalizedX = (sourcePoint.x - transformedBounds.minX) / transformedBounds.width
        let normalizedY = (sourcePoint.y - transformedBounds.minY) / transformedBounds.height

        return SwingPoint(
            x: Double(min(max(normalizedX, 0), 1)),
            y: Double(min(max(normalizedY, 0), 1))
        )
    }

    private static func safeSize(_ size: CGSize) -> CGSize {
        CGSize(width: max(1, abs(size.width)), height: max(1, abs(size.height)))
    }

    private static func aspectRatiosMatch(_ lhs: CGSize, _ rhs: CGSize, tolerance: CGFloat = 0.03) -> Bool {
        let lhsRatio = lhs.width / lhs.height
        let rhsRatio = rhs.width / rhs.height
        return abs(lhsRatio - rhsRatio) <= tolerance
    }
}
