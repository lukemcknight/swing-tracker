import SwiftUI

private let bodySegments: [(String, String)] = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

private let footSides = ["left", "right"]
private let bodyBoxKeypointNames = [
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

private let overlayGreen = Color(red: 0.12, green: 0.82, blue: 0.34)
private let overlayBlue = Color(red: 0.12, green: 0.18, blue: 0.96)
private let overlayAmber = Color(red: 1.0, green: 0.72, blue: 0.16)
private let overlayRedJoint = Color(red: 1.0, green: 0.26, blue: 0.26)
private let overlayBlueJoint = Color(red: 0.16, green: 0.32, blue: 0.92)

private enum ClubPathTracePhase {
    case backswing
    case transition
    case downswing
    case followThrough

    var color: Color {
        switch self {
        case .backswing:
            Color(red: 0.55, green: 0.92, blue: 0.28)
        case .transition:
            overlayAmber
        case .downswing:
            Color(red: 1.0, green: 0.22, blue: 0.14)
        case .followThrough:
            Color(red: 0.13, green: 0.78, blue: 0.70)
        }
    }
}

struct SkeletonOverlay: View {
    let frame: SwingAnalysisFrame?
    let trailFrames: [SwingAnalysisFrame]
    let clubPathFrames: [SwingAnalysisFrame]
    let coordinateSpace: VideoCoordinateSpace
    let selectedOverlays: Set<AnalysisOverlay>
    let analysis: SwingAnalysis
    private let reliableScore = 0.35
    private let drawableScore = 0.2
    private let inferredScoreCap = 0.42
    private let footAxisAlignmentThreshold = 0.55

    var body: some View {
        Canvas { context, size in
            guard let frame else { return }
            let rect = CGRect(origin: .zero, size: size)
            let keypoints = repairedDisplayKeypoints(
                Dictionary(uniqueKeysWithValues: frame.keypoints.map { ($0.name, $0) })
            )

            if selectedOverlays.contains(.guides) {
                drawGuides(context: &context, rect: rect)
            }
            if selectedOverlays.contains(.swingPlane) {
                drawSwingPlane(context: &context, rect: rect)
            }
            if selectedOverlays.contains(.bodyBox) {
                drawBodyBox(context: &context, rect: rect, keypoints: keypoints)
            }
            if selectedOverlays.contains(.wristPath) {
                drawWristTrail(context: &context, rect: rect)
            }
            if selectedOverlays.contains(.headTracking) {
                drawHeadTracking(context: &context, rect: rect, currentKeypoints: keypoints)
            }
            if selectedOverlays.contains(.clubPath) {
                drawClubPath(context: &context, rect: rect)
                drawBallMarker(context: &context, rect: rect)
            }
            if selectedOverlays.contains(.spineTracking) {
                drawSpineTracking(context: &context, rect: rect, keypoints: keypoints)
            }
            if selectedOverlays.contains(.joints) {
                drawSegments(context: &context, rect: rect, keypoints: keypoints)
                drawFootSegments(context: &context, rect: rect, keypoints: keypoints)
                drawPoints(context: &context, rect: rect, keypoints: keypoints)
            }
        }
        .allowsHitTesting(false)
    }

    private func drawGuides(context: inout GraphicsContext, rect: CGRect) {
        guard let address = analysis.frame(for: .address) else { return }
        let keypoints = repairedDisplayKeypoints(
            Dictionary(uniqueKeysWithValues: address.keypoints.map { ($0.name, $0) })
        )
        let guideColor = overlayGreen.opacity(0.86)

        drawAddressBodyGate(context: &context, rect: rect, keypoints: keypoints, color: guideColor)

        if let nose = keypoint("nose", in: keypoints, minimumScore: drawableScore) {
            let bodyHeight = referenceBodyHeight(in: keypoints)
            let shoulderWidth = normalizedDistance(
                firstName: "left_shoulder",
                secondName: "right_shoulder",
                keypoints: keypoints
            ) ?? bodyHeight * 0.24
            let halfWidth = max(0.035, min(0.10, shoulderWidth * 0.50))
            let halfHeight = max(0.025, min(0.075, bodyHeight * 0.09))
            drawNormalizedRect(
                context: &context,
                center: nose,
                halfWidth: halfWidth,
                halfHeight: halfHeight,
                rect: rect,
                color: guideColor
            )
        }

        if
            let shoulders = midpoint(firstName: "left_shoulder", secondName: "right_shoulder", keypoints: keypoints),
            let hips = midpoint(firstName: "left_hip", secondName: "right_hip", keypoints: keypoints)
        {
            let start = mappedPoint(shoulders, into: rect)
            let end = mappedPoint(hips, into: rect)
            var path = Path()
            path.move(to: start)
            path.addLine(to: end)
            context.stroke(
                path,
                with: .color(guideColor),
                style: StrokeStyle(lineWidth: 3, lineCap: .round, dash: [8, 7])
            )
        }

        if let stance = stanceGuidePoints(in: keypoints) {
            let first = mappedPoint(stance.first, into: rect)
            let second = mappedPoint(stance.second, into: rect)
            let endpoints = extendedLineEndpoints(through: first, and: second, in: rect)
            var path = Path()
            path.move(to: endpoints.start)
            path.addLine(to: endpoints.end)
            context.stroke(
                path,
                with: .color(guideColor.opacity(0.78)),
                style: StrokeStyle(lineWidth: 3, lineCap: .round, dash: [12, 8])
            )
        }
    }

    private func drawAddressBodyGate(
        context: inout GraphicsContext,
        rect: CGRect,
        keypoints: [String: SwingKeypoint],
        color: Color
    ) {
        let points = bodyBoxKeypointNames
            .compactMap { keypoints[$0] }
            .filter { $0.score >= drawableScore }
        guard points.count >= 3 else { return }

        let minX = points.map(\.x).min() ?? 0
        let maxX = points.map(\.x).max() ?? 0
        let minY = points.map(\.y).min() ?? 0
        let maxY = points.map(\.y).max() ?? 0
        guard maxX > minX, maxY > minY else { return }

        let topLeft = mappedPoint(
            SwingPoint(x: max(0, minX - 0.02), y: max(0, minY - 0.02)),
            into: rect
        )
        let bottomRight = mappedPoint(
            SwingPoint(x: min(1, maxX + 0.02), y: min(1, maxY + 0.02)),
            into: rect
        )
        let gate = CGRect(
            x: min(topLeft.x, bottomRight.x),
            y: min(topLeft.y, bottomRight.y),
            width: abs(bottomRight.x - topLeft.x),
            height: abs(bottomRight.y - topLeft.y)
        )
        guard gate.width > 8, gate.height > 8 else { return }

        var path = Path()
        path.addRect(gate)
        context.stroke(path, with: .color(color.opacity(0.88)), lineWidth: 3)
    }

    private func drawSwingPlane(context: inout GraphicsContext, rect: CGRect) {
        guard let plane = swingPlaneFit(in: rect) else { return }
        let planeColor = overlayBlue
        let normal = CGPoint(x: -plane.direction.y, y: plane.direction.x)
        let halfBand = plane.bandWidth / 2

        var band = Path()
        band.move(to: CGPoint(x: plane.start.x + normal.x * halfBand, y: plane.start.y + normal.y * halfBand))
        band.addLine(to: CGPoint(x: plane.end.x + normal.x * halfBand, y: plane.end.y + normal.y * halfBand))
        band.addLine(to: CGPoint(x: plane.end.x - normal.x * halfBand, y: plane.end.y - normal.y * halfBand))
        band.addLine(to: CGPoint(x: plane.start.x - normal.x * halfBand, y: plane.start.y - normal.y * halfBand))
        band.closeSubpath()
        context.fill(band, with: .color(planeColor.opacity(0.16)))

        var centerLine = Path()
        centerLine.move(to: plane.start)
        centerLine.addLine(to: plane.end)
        context.stroke(centerLine, with: .color(.black.opacity(0.46)), lineWidth: 10)
        context.stroke(centerLine, with: .color(planeColor.opacity(0.82)), lineWidth: 5)
    }

    private func drawWristTrail(context: inout GraphicsContext, rect: CGRect) {
        for (index, trailFrame) in trailFrames.enumerated() {
            guard let point = wristCenter(in: trailFrame) else { continue }
            let opacity = Double(index + 1) / Double(max(1, trailFrames.count))
            let mapped = mappedPoint(point, into: rect)
            let circle = CGRect(x: mapped.x - 4, y: mapped.y - 4, width: 8, height: 8)
            context.fill(Path(ellipseIn: circle), with: .color(Theme.accent.opacity(opacity)))
        }
    }

    private func drawHeadTracking(
        context: inout GraphicsContext,
        rect: CGRect,
        currentKeypoints: [String: SwingKeypoint]
    ) {
        let headColor = Color(red: 0.13, green: 0.74, blue: 0.95)
        for (index, trailFrame) in trailFrames.enumerated() {
            guard let point = keypoint("nose", in: trailFrame, minimumScore: drawableScore) else { continue }
            let opacity = Double(index + 1) / Double(max(1, trailFrames.count))
            let mapped = mappedPoint(point, into: rect)
            let circle = CGRect(x: mapped.x - 3.5, y: mapped.y - 3.5, width: 7, height: 7)
            context.fill(Path(ellipseIn: circle), with: .color(headColor.opacity(opacity)))
        }

        guard let head = currentKeypoints["nose"], head.score >= drawableScore else { return }
        let mapped = mappedPoint(SwingPoint(x: head.x, y: head.y), into: rect)
        let marker = CGRect(x: mapped.x - 12, y: mapped.y - 12, width: 24, height: 24)
        context.stroke(Path(ellipseIn: marker), with: .color(headColor.opacity(0.95)), lineWidth: 3)
        context.fill(
            Path(ellipseIn: marker.insetBy(dx: 7, dy: 7)),
            with: .color(.white.opacity(0.92))
        )
    }

    private func drawClubPath(context: inout GraphicsContext, rect: CGRect) {
        let segments = ClubPathStabilizer.stableSegments(from: clubPathFrames)
        guard !segments.isEmpty else { return }

        for segment in segments {
            for run in clubPathPhaseRuns(from: segment.samples) {
                drawClubPathRun(
                    context: &context,
                    rect: rect,
                    samples: run.samples,
                    color: run.phase.color
                )
            }

            if let finalSample = segment.samples.last {
                drawClubPathEndpoint(
                    context: &context,
                    rect: rect,
                    sample: finalSample,
                    color: clubPathTracePhase(for: finalSample.frameIndex).color
                )
            }
        }
    }

    private func clubPathPhaseRuns(
        from samples: [ClubPathSample]
    ) -> [(phase: ClubPathTracePhase, samples: [ClubPathSample])] {
        guard let first = samples.first else { return [] }

        var runs: [(phase: ClubPathTracePhase, samples: [ClubPathSample])] = []
        var currentPhase = clubPathTracePhase(for: first.frameIndex)
        var currentSamples = [first]
        var previousSample = first

        for sample in samples.dropFirst() {
            let phase = clubPathTracePhase(for: sample.frameIndex)
            if phase == currentPhase {
                currentSamples.append(sample)
            } else {
                if currentSamples.count >= 2 {
                    runs.append((currentPhase, currentSamples))
                }
                currentPhase = phase
                currentSamples = [previousSample, sample]
            }
            previousSample = sample
        }

        if currentSamples.count >= 2 {
            runs.append((currentPhase, currentSamples))
        }

        return runs
    }

    private func clubPathTracePhase(for frameIndex: Int) -> ClubPathTracePhase {
        let topIndex = analysis.frameIndex(for: .top)
        let impactIndex = analysis.frameIndex(for: .impact)

        if let topIndex, frameIndex <= topIndex {
            return .backswing
        }
        if let impactIndex, frameIndex > impactIndex {
            return .followThrough
        }
        if
            let topIndex,
            let impactIndex,
            impactIndex > topIndex,
            frameIndex <= impactIndex
        {
            let transitionEnd = topIndex + max(1, Int(Double(impactIndex - topIndex) * 0.35))
            return frameIndex <= transitionEnd ? .transition : .downswing
        }

        return .downswing
    }

    private func drawClubPathRun(
        context: inout GraphicsContext,
        rect: CGRect,
        samples: [ClubPathSample],
        color: Color
    ) {
        let mappedPoints = samples.map { mappedPoint($0.point, into: rect) }
        guard mappedPoints.count >= 2 else { return }

        let path = smoothedPath(through: mappedPoints)
        context.stroke(
            path,
            with: .color(.black.opacity(0.52)),
            style: StrokeStyle(lineWidth: 9, lineCap: .round, lineJoin: .round)
        )
        context.stroke(
            path,
            with: .color(color.opacity(0.9)),
            style: StrokeStyle(lineWidth: 5.5, lineCap: .round, lineJoin: .round)
        )
    }

    private func drawClubPathEndpoint(
        context: inout GraphicsContext,
        rect: CGRect,
        sample: ClubPathSample,
        color: Color
    ) {
        let point = mappedPoint(sample.point, into: rect)
        let shadow = CGRect(x: point.x - 8, y: point.y - 8, width: 16, height: 16)
        let marker = CGRect(x: point.x - 5, y: point.y - 5, width: 10, height: 10)
        context.fill(Path(ellipseIn: shadow), with: .color(.black.opacity(0.42)))
        context.fill(Path(ellipseIn: marker), with: .color(color.opacity(0.98)))
        context.stroke(Path(ellipseIn: marker), with: .color(.white.opacity(0.45)), lineWidth: 1.5)
    }

    private func smoothedPath(through points: [CGPoint]) -> Path {
        var path = Path()
        guard let first = points.first else { return path }

        path.move(to: first)
        guard points.count > 2 else {
            if let last = points.last {
                path.addLine(to: last)
            }
            return path
        }

        var previous = first
        for point in points.dropFirst() {
            let midpoint = CGPoint(
                x: (previous.x + point.x) / 2,
                y: (previous.y + point.y) / 2
            )
            path.addQuadCurve(to: midpoint, control: previous)
            previous = point
        }
        if let last = points.last {
            path.addLine(to: last)
        }

        return path
    }

    private func drawBallMarker(context: inout GraphicsContext, rect: CGRect) {
        guard let ball = estimatedBallPosition() else { return }

        let mapped = mappedPoint(ball, into: rect)
        let shadow = CGRect(x: mapped.x - 13, y: mapped.y - 13, width: 26, height: 26)
        let marker = CGRect(x: mapped.x - 8, y: mapped.y - 8, width: 16, height: 16)
        context.fill(Path(ellipseIn: shadow), with: .color(.black.opacity(0.36)))
        context.fill(Path(ellipseIn: marker), with: .color(.white.opacity(0.96)))
        context.stroke(Path(ellipseIn: marker), with: .color(.black.opacity(0.30)), lineWidth: 2.5)
    }

    private func estimatedBallPosition() -> SwingPoint? {
        if
            let addressIndex = analysis.frameIndex(for: .address),
            let ball = nearestClubHead(around: addressIndex, radius: 3)
        {
            return ball
        }

        if
            let addressIndex = analysis.frameIndex(for: .address),
            let takeawayIndex = analysis.frameIndex(for: .takeaway),
            let ball = groundedClubHead(from: addressIndex, through: takeawayIndex)
        {
            return ball
        }

        return nil
    }

    private func nearestClubHead(around index: Int, radius: Int) -> SwingPoint? {
        guard !analysis.frames.isEmpty else { return nil }

        let lowerBound = max(0, index - radius)
        let upperBound = min(analysis.frames.count - 1, index + radius)
        return analysis.primaryClubPathSegment(in: lowerBound...upperBound)?
            .samples
            .map { sample in (distance: abs(sample.frameIndex - index), point: sample.point) }
            .min { $0.distance < $1.distance }?
            .point
    }

    private func groundedClubHead(from startIndex: Int, through endIndex: Int) -> SwingPoint? {
        guard !analysis.frames.isEmpty else { return nil }

        let boundedStart = min(max(0, startIndex), analysis.frames.count - 1)
        let boundedEnd = min(max(0, endIndex), analysis.frames.count - 1)
        guard boundedStart <= boundedEnd else { return nil }

        let earlyPoints = analysis.primaryClubPathSegment(in: boundedStart...boundedEnd)?.points ?? []
        guard !earlyPoints.isEmpty else { return nil }

        let sortedByHeight = earlyPoints.sorted { $0.y > $1.y }
        return sortedByHeight.prefix(3).min { $0.x < $1.x } ?? sortedByHeight.first
    }

    private func drawSpineTracking(
        context: inout GraphicsContext,
        rect: CGRect,
        keypoints: [String: SwingKeypoint]
    ) {
        guard
            let shoulders = midpoint(firstName: "left_shoulder", secondName: "right_shoulder", keypoints: keypoints),
            let hips = midpoint(firstName: "left_hip", secondName: "right_hip", keypoints: keypoints)
        else { return }

        let start = mappedPoint(shoulders, into: rect)
        let end = mappedPoint(hips, into: rect)
        let spineColor = overlayGreen
        var path = Path()
        path.move(to: start)
        path.addLine(to: end)
        context.stroke(path, with: .color(.black.opacity(0.56)), lineWidth: 8)
        context.stroke(path, with: .color(spineColor.opacity(0.95)), lineWidth: 4)

        for point in [start, end] {
            let circle = CGRect(x: point.x - 5, y: point.y - 5, width: 10, height: 10)
            context.fill(Path(ellipseIn: circle), with: .color(.white.opacity(0.92)))
            context.stroke(Path(ellipseIn: circle), with: .color(spineColor), lineWidth: 2)
        }
    }

    private func drawBodyBox(
        context: inout GraphicsContext,
        rect: CGRect,
        keypoints: [String: SwingKeypoint]
    ) {
        let mappedPoints = bodyBoxKeypointNames
            .compactMap { keypoints[$0] }
            .filter { $0.score >= reliableScore }
            .map { mappedPoint(SwingPoint(x: $0.x, y: $0.y), into: rect) }

        guard mappedPoints.count >= 3 else { return }

        let minX = mappedPoints.map(\.x).min() ?? rect.minX
        let maxX = mappedPoints.map(\.x).max() ?? rect.maxX
        let minY = mappedPoints.map(\.y).min() ?? rect.minY
        let maxY = mappedPoints.map(\.y).max() ?? rect.maxY
        var box = CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
            .insetBy(dx: -12, dy: -12)
            .intersection(rect)
        box = box.standardized
        guard box.width > 4, box.height > 4 else { return }

        let boxPath = Path(roundedRect: box, cornerRadius: 12)
        context.fill(boxPath, with: .color(overlayGreen.opacity(0.06)))
        context.stroke(boxPath, with: .color(overlayGreen.opacity(0.96)), lineWidth: 3)
    }

    private func drawSegments(
        context: inout GraphicsContext,
        rect: CGRect,
        keypoints: [String: SwingKeypoint]
    ) {
        for segment in bodySegments {
            guard
                let first = keypoints[segment.0],
                let second = keypoints[segment.1],
                first.score > 0.2,
                second.score > 0.2
            else { continue }

            let start = mappedPoint(SwingPoint(x: first.x, y: first.y), into: rect)
            let end = mappedPoint(SwingPoint(x: second.x, y: second.y), into: rect)
            var path = Path()
            path.move(to: start)
            path.addLine(to: end)
            context.stroke(path, with: .color(.black.opacity(0.42)), lineWidth: 8)
            context.stroke(path, with: .color(overlayGreen.opacity(0.78)), lineWidth: 5)
        }
    }

    private func drawFootSegments(
        context: inout GraphicsContext,
        rect: CGRect,
        keypoints: [String: SwingKeypoint]
    ) {
        for side in footSides {
            guard
                let heel = keypoints["\(side)_heel"],
                let toe = keypoints["\(side)_foot_index"],
                heel.score > drawableScore,
                toe.score > drawableScore
            else { continue }

            let start = mappedPoint(SwingPoint(x: heel.x, y: heel.y), into: rect)
            let end = mappedPoint(SwingPoint(x: toe.x, y: toe.y), into: rect)
            var path = Path()
            path.move(to: start)
            path.addLine(to: end)
            context.stroke(path, with: .color(.black.opacity(0.42)), lineWidth: 8)
            context.stroke(path, with: .color(overlayGreen.opacity(0.78)), lineWidth: 5)
        }
    }

    private func drawPoints(
        context: inout GraphicsContext,
        rect: CGRect,
        keypoints: [String: SwingKeypoint]
    ) {
        for point in keypoints.values where point.score > 0.2 {
            let mapped = mappedPoint(SwingPoint(x: point.x, y: point.y), into: rect)
            let radius: CGFloat = point.name.contains("wrist") ? 6 : 5
            let circle = CGRect(x: mapped.x - radius, y: mapped.y - radius, width: radius * 2, height: radius * 2)
            context.fill(Path(ellipseIn: circle.insetBy(dx: -2, dy: -2)), with: .color(.black.opacity(0.32)))
            context.fill(Path(ellipseIn: circle), with: .color(jointColor(for: point.name).opacity(0.95)))
            context.stroke(Path(ellipseIn: circle), with: .color(.white.opacity(0.22)), lineWidth: 1.5)
        }
    }

    private func jointColor(for name: String) -> Color {
        if name.contains("left") {
            return overlayRedJoint
        }
        if name.contains("right") {
            return overlayBlueJoint
        }
        return .white
    }

    private func drawConnectedTrail(
        context: inout GraphicsContext,
        rect: CGRect,
        points: [SwingPoint],
        color: Color,
        radius: CGFloat,
        lineWidth: CGFloat
    ) {
        guard !points.isEmpty else { return }

        let mappedPoints = points.map { mappedPoint($0, into: rect) }
        if mappedPoints.count >= 2 {
            var path = Path()
            path.move(to: mappedPoints[0])
            for point in mappedPoints.dropFirst() {
                path.addLine(to: point)
            }
            context.stroke(path, with: .color(color.opacity(0.72)), lineWidth: lineWidth)
        }

        for (index, point) in mappedPoints.enumerated() {
            let opacity = Double(index + 1) / Double(max(1, mappedPoints.count))
            let circle = CGRect(
                x: point.x - radius,
                y: point.y - radius,
                width: radius * 2,
                height: radius * 2
            )
            context.fill(Path(ellipseIn: circle), with: .color(color.opacity(opacity)))
        }
    }

    private func mappedPoint(_ point: SwingPoint, into rect: CGRect) -> CGPoint {
        let displayPoint = coordinateSpace.mapNormalizedPoint(x: point.x, y: point.y)
        return VideoGeometry.mapNormalizedPoint(x: displayPoint.x, y: displayPoint.y, into: rect)
    }

    private func keypoint(
        _ name: String,
        in keypoints: [String: SwingKeypoint],
        minimumScore: Double
    ) -> SwingPoint? {
        guard let point = keypoints[name], point.score >= minimumScore else {
            return nil
        }
        return SwingPoint(x: point.x, y: point.y)
    }

    private func keypoint(
        _ name: String,
        in frame: SwingAnalysisFrame,
        minimumScore: Double
    ) -> SwingPoint? {
        guard let point = frame.keypoints.first(where: { $0.name == name }), point.score >= minimumScore else {
            return nil
        }
        return SwingPoint(x: point.x, y: point.y)
    }

    private func drawNormalizedRect(
        context: inout GraphicsContext,
        center: SwingPoint,
        halfWidth: Double,
        halfHeight: Double,
        rect: CGRect,
        color: Color
    ) {
        let topLeft = mappedPoint(
            SwingPoint(x: center.x - halfWidth, y: center.y - halfHeight),
            into: rect
        )
        let bottomRight = mappedPoint(
            SwingPoint(x: center.x + halfWidth, y: center.y + halfHeight),
            into: rect
        )
        let guideRect = CGRect(
            x: min(topLeft.x, bottomRight.x),
            y: min(topLeft.y, bottomRight.y),
            width: abs(bottomRight.x - topLeft.x),
            height: abs(bottomRight.y - topLeft.y)
        )
        guard guideRect.width > 4, guideRect.height > 4 else { return }

        let path = Path(roundedRect: guideRect, cornerRadius: 8)
        context.fill(path, with: .color(color.opacity(0.08)))
        context.stroke(path, with: .color(color), style: StrokeStyle(lineWidth: 3, lineCap: .round, dash: [8, 7]))
    }

    private func stanceGuidePoints(in keypoints: [String: SwingKeypoint]) -> (first: SwingPoint, second: SwingPoint)? {
        if
            let left = keypoint("left_ankle", in: keypoints, minimumScore: drawableScore),
            let right = keypoint("right_ankle", in: keypoints, minimumScore: drawableScore)
        {
            return (left, right)
        }
        if
            let left = keypoint("left_heel", in: keypoints, minimumScore: drawableScore),
            let right = keypoint("right_heel", in: keypoints, minimumScore: drawableScore)
        {
            return (left, right)
        }
        return nil
    }

    private func extendedLineEndpoints(
        through first: CGPoint,
        and second: CGPoint,
        in rect: CGRect
    ) -> (start: CGPoint, end: CGPoint) {
        let dx = second.x - first.x
        let dy = second.y - first.y
        let length = max(1, hypot(dx, dy))
        let unit = CGPoint(x: dx / length, y: dy / length)
        let center = CGPoint(x: (first.x + second.x) / 2, y: (first.y + second.y) / 2)
        let extent = max(rect.width, rect.height) * 1.5
        return (
            CGPoint(x: center.x - unit.x * extent, y: center.y - unit.y * extent),
            CGPoint(x: center.x + unit.x * extent, y: center.y + unit.y * extent)
        )
    }

    private func swingPlaneFit(in rect: CGRect) -> (start: CGPoint, end: CGPoint, direction: CGPoint, bandWidth: CGFloat)? {
        let points = swingPlaneCandidateFrames()
            .compactMap(\.clubHead)
            .map { mappedPoint($0, into: rect) }
        guard points.count >= 3 else { return nil }

        let meanX = points.map(\.x).reduce(0, +) / CGFloat(points.count)
        let meanY = points.map(\.y).reduce(0, +) / CGFloat(points.count)
        let centered = points.map { CGPoint(x: $0.x - meanX, y: $0.y - meanY) }
        let covarianceXX = centered.map { $0.x * $0.x }.reduce(0, +) / CGFloat(points.count)
        let covarianceYY = centered.map { $0.y * $0.y }.reduce(0, +) / CGFloat(points.count)
        let covarianceXY = centered.map { $0.x * $0.y }.reduce(0, +) / CGFloat(points.count)
        let angle = 0.5 * atan2(2 * covarianceXY, covarianceXX - covarianceYY)
        let direction = CGPoint(x: cos(angle), y: sin(angle))
        let projections = centered.map { $0.x * direction.x + $0.y * direction.y }
        guard
            let minProjection = projections.min(),
            let maxProjection = projections.max(),
            maxProjection - minProjection >= 8
        else { return nil }

        let residuals = centered.map { abs($0.x * direction.y - $0.y * direction.x) }.sorted()
        let medianResidual = residuals[residuals.count / 2]
        let bandWidth = max(10, min(42, medianResidual * 2.4))
        let extensionAmount = max(10, (maxProjection - minProjection) * 0.10)
        let center = CGPoint(x: meanX, y: meanY)
        let start = CGPoint(
            x: center.x + direction.x * (minProjection - extensionAmount),
            y: center.y + direction.y * (minProjection - extensionAmount)
        )
        let end = CGPoint(
            x: center.x + direction.x * (maxProjection + extensionAmount),
            y: center.y + direction.y * (maxProjection + extensionAmount)
        )
        return (start, end, direction, bandWidth)
    }

    private func swingPlaneCandidateFrames() -> [SwingAnalysisFrame] {
        let framesWithClubHead = analysis.primaryClubPathSegment()?.frames ?? []
        guard
            let startIndex = analysis.frameIndex(for: .takeaway),
            let endIndex = analysis.frameIndex(for: .impact),
            endIndex >= startIndex
        else {
            return framesWithClubHead
        }

        let phaseFrames = analysis.primaryClubPathSegment(in: startIndex...endIndex)?.frames ?? []
        return phaseFrames.count >= 3 ? phaseFrames : framesWithClubHead
    }

    private func midpoint(
        firstName: String,
        secondName: String,
        keypoints: [String: SwingKeypoint]
    ) -> SwingPoint? {
        guard
            let first = keypoints[firstName],
            let second = keypoints[secondName],
            first.score >= drawableScore,
            second.score >= drawableScore
        else { return nil }

        let firstWeight = max(0.05, first.score)
        let secondWeight = max(0.05, second.score)
        let totalWeight = firstWeight + secondWeight
        return SwingPoint(
            x: (first.x * firstWeight + second.x * secondWeight) / totalWeight,
            y: (first.y * firstWeight + second.y * secondWeight) / totalWeight
        )
    }

    private func normalizedDistance(
        firstName: String,
        secondName: String,
        keypoints: [String: SwingKeypoint]
    ) -> Double? {
        guard
            let first = keypoints[firstName],
            let second = keypoints[secondName],
            first.score >= drawableScore,
            second.score >= drawableScore
        else { return nil }

        return hypot(second.x - first.x, second.y - first.y)
    }

    private func wristCenter(in frame: SwingAnalysisFrame) -> SwingPoint? {
        let points = repairedOccludedArmKeypoints(
            Dictionary(uniqueKeysWithValues: frame.keypoints.map { ($0.name, $0) })
        )
        guard let left = points["left_wrist"], let right = points["right_wrist"] else {
            return nil
        }

        if left.score >= reliableScore && right.score >= reliableScore {
            let leftWeight = max(0.05, left.score)
            let rightWeight = max(0.05, right.score)
            let totalWeight = leftWeight + rightWeight
            return SwingPoint(
                x: (left.x * leftWeight + right.x * rightWeight) / totalWeight,
                y: (left.y * leftWeight + right.y * rightWeight) / totalWeight
            )
        }
        if left.score >= reliableScore {
            return SwingPoint(x: left.x, y: left.y)
        }
        if right.score >= reliableScore {
            return SwingPoint(x: right.x, y: right.y)
        }
        if left.score >= drawableScore && right.score >= drawableScore {
            let leftWeight = max(0.05, left.score)
            let rightWeight = max(0.05, right.score)
            let totalWeight = leftWeight + rightWeight
            return SwingPoint(
                x: (left.x * leftWeight + right.x * rightWeight) / totalWeight,
                y: (left.y * leftWeight + right.y * rightWeight) / totalWeight
            )
        }
        if left.score >= drawableScore {
            return SwingPoint(x: left.x, y: left.y)
        }
        if right.score >= drawableScore {
            return SwingPoint(x: right.x, y: right.y)
        }

        return SwingPoint(x: (left.x + right.x) / 2, y: (left.y + right.y) / 2)
    }

    private func repairedOccludedArmKeypoints(_ keypoints: [String: SwingKeypoint]) -> [String: SwingKeypoint] {
        var repaired = keypoints
        inferOccludedArm(hiddenSide: "left", visibleSide: "right", keypoints: &repaired)
        inferOccludedArm(hiddenSide: "right", visibleSide: "left", keypoints: &repaired)
        return repaired
    }

    private func repairedDisplayKeypoints(_ keypoints: [String: SwingKeypoint]) -> [String: SwingKeypoint] {
        var repaired = repairedOccludedArmKeypoints(keypoints)
        repairFoot(side: "left", keypoints: &repaired)
        repairFoot(side: "right", keypoints: &repaired)
        return repaired
    }

    private func inferOccludedArm(
        hiddenSide: String,
        visibleSide: String,
        keypoints: inout [String: SwingKeypoint]
    ) {
        let hiddenShoulderName = "\(hiddenSide)_shoulder"
        let hiddenElbowName = "\(hiddenSide)_elbow"
        let hiddenWristName = "\(hiddenSide)_wrist"
        let visibleWristName = "\(visibleSide)_wrist"

        guard
            let hiddenShoulder = keypoints[hiddenShoulderName],
            let visibleWrist = keypoints[visibleWristName],
            hiddenShoulder.score >= reliableScore,
            visibleWrist.score >= reliableScore
        else { return }

        let shoulderToGrip = hypot(visibleWrist.x - hiddenShoulder.x, visibleWrist.y - hiddenShoulder.y)
        guard shoulderToGrip >= 0.04, shoulderToGrip <= 0.85 else { return }

        let inferredScore = min(
            inferredScoreCap,
            max(drawableScore + 0.03, min(hiddenShoulder.score, visibleWrist.score) * 0.62)
        )

        if let hiddenWrist = keypoints[hiddenWristName], hiddenWrist.score < reliableScore {
            keypoints[hiddenWristName] = SwingKeypoint(
                name: hiddenWristName,
                x: visibleWrist.x,
                y: visibleWrist.y,
                score: inferredScore
            )
        }

        if let hiddenElbow = keypoints[hiddenElbowName], hiddenElbow.score < reliableScore {
            keypoints[hiddenElbowName] = SwingKeypoint(
                name: hiddenElbowName,
                x: hiddenShoulder.x + (visibleWrist.x - hiddenShoulder.x) * 0.58,
                y: hiddenShoulder.y + (visibleWrist.y - hiddenShoulder.y) * 0.58,
                score: inferredScore
            )
        }
    }

    private func repairFoot(side: String, keypoints: inout [String: SwingKeypoint]) {
        let bodyHeight = referenceBodyHeight(in: keypoints)
        for name in ["\(side)_heel", "\(side)_foot_index"] {
            guard let point = keypoints[name] else { continue }
            if !isPlausibleFootPoint(point, side: side, bodyHeight: bodyHeight, keypoints: keypoints) {
                keypoints[name] = suppressedKeypoint(point)
            }
        }
        repairFootAxis(side: side, bodyHeight: bodyHeight, keypoints: &keypoints)
    }

    private func repairFootAxis(side: String, bodyHeight: Double, keypoints: inout [String: SwingKeypoint]) {
        guard
            let heel = keypoints["\(side)_heel"],
            let toe = keypoints["\(side)_foot_index"],
            let ankle = keypoints["\(side)_ankle"]
        else { return }

        if ankle.score < drawableScore {
            keypoints["\(side)_heel"] = suppressedKeypoint(heel)
            keypoints["\(side)_foot_index"] = suppressedKeypoint(toe)
            return
        }
        guard
            isPlausibleFootPoint(heel, side: side, bodyHeight: bodyHeight, keypoints: keypoints),
            isPlausibleFootPoint(toe, side: side, bodyHeight: bodyHeight, keypoints: keypoints)
        else {
            keypoints["\(side)_heel"] = suppressedKeypoint(heel)
            keypoints["\(side)_foot_index"] = suppressedKeypoint(toe)
            return
        }

        guard let footAxis = normalizedVector(from: heel, to: toe) else {
            synthesizeStanceAlignedFoot(side: side, bodyHeight: bodyHeight, keypoints: &keypoints)
            return
        }

        let stanceAxis = stanceAxis(in: keypoints)
        if abs(dotProduct(footAxis, stanceAxis)) < footAxisAlignmentThreshold {
            synthesizeStanceAlignedFoot(side: side, bodyHeight: bodyHeight, keypoints: &keypoints)
        }
    }

    private func synthesizeStanceAlignedFoot(
        side: String,
        bodyHeight: Double,
        keypoints: inout [String: SwingKeypoint]
    ) {
        guard
            let heel = keypoints["\(side)_heel"],
            let toe = keypoints["\(side)_foot_index"]
        else { return }

        var axis = stanceAxis(in: keypoints)
        if let rawAxis = normalizedVector(from: heel, to: toe), dotProduct(rawAxis, axis) < 0 {
            axis = (x: -axis.x, y: -axis.y)
        }

        let rawLength = hypot(toe.x - heel.x, toe.y - heel.y)
        let minLength = max(0.035, bodyHeight * 0.07)
        let maxLength = max(minLength, min(0.16, bodyHeight * 0.22))
        let footLength = max(minLength, min(maxLength, rawLength))
        let halfLength = footLength / 2
        let center = (x: (heel.x + toe.x) / 2, y: (heel.y + toe.y) / 2)
        let inferredScore = min(
            inferredScoreCap,
            max(drawableScore + 0.03, min(heel.score, toe.score) * 0.62)
        )

        let candidateHeel = SwingKeypoint(
            name: heel.name,
            x: center.x - axis.x * halfLength,
            y: center.y - axis.y * halfLength,
            score: inferredScore
        )
        let candidateToe = SwingKeypoint(
            name: toe.name,
            x: center.x + axis.x * halfLength,
            y: center.y + axis.y * halfLength,
            score: inferredScore
        )
        guard
            isPlausibleFootPoint(candidateHeel, side: side, bodyHeight: bodyHeight, keypoints: keypoints),
            isPlausibleFootPoint(candidateToe, side: side, bodyHeight: bodyHeight, keypoints: keypoints)
        else {
            keypoints["\(side)_heel"] = suppressedKeypoint(heel)
            keypoints["\(side)_foot_index"] = suppressedKeypoint(toe)
            return
        }

        keypoints["\(side)_heel"] = candidateHeel
        keypoints["\(side)_foot_index"] = candidateToe
    }

    private func isPlausibleFootPoint(
        _ point: SwingKeypoint,
        side: String,
        bodyHeight: Double,
        keypoints: [String: SwingKeypoint]
    ) -> Bool {
        guard let ankle = keypoints["\(side)_ankle"] else { return false }
        guard point.score >= drawableScore, ankle.score >= drawableScore else { return false }
        guard (-0.15...1.15).contains(point.x), (-0.1...1.15).contains(point.y) else { return false }

        let maxDistance = max(0.055, min(0.22, bodyHeight * 0.30))
        let maxVerticalDelta = max(0.055, min(0.18, bodyHeight * 0.24))
        let distanceFromAnkle = hypot(point.x - ankle.x, point.y - ankle.y)

        return distanceFromAnkle <= maxDistance && abs(point.y - ankle.y) <= maxVerticalDelta
    }

    private func referenceBodyHeight(in keypoints: [String: SwingKeypoint]) -> Double {
        let names = [
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_wrist",
            "right_wrist",
        ]
        let points = names.compactMap { keypoints[$0] }.filter { $0.score >= drawableScore }
        guard !points.isEmpty else { return 0.55 }

        let minY = points.map(\.y).min() ?? 0
        let maxY = points.map(\.y).max() ?? 0
        return max(0.22, min(1.05, maxY - minY))
    }

    private func suppressedKeypoint(_ point: SwingKeypoint) -> SwingKeypoint {
        SwingKeypoint(
            name: point.name,
            x: point.x,
            y: point.y,
            score: min(point.score, drawableScore - 0.01)
        )
    }

    private func stanceAxis(in keypoints: [String: SwingKeypoint]) -> (x: Double, y: Double) {
        if let ankleAxis = normalizedPairAxis(firstName: "left_ankle", secondName: "right_ankle", keypoints: keypoints) {
            return ankleAxis
        }
        if let hipAxis = normalizedPairAxis(firstName: "left_hip", secondName: "right_hip", keypoints: keypoints) {
            return hipAxis
        }
        return (x: 1, y: 0)
    }

    private func normalizedPairAxis(
        firstName: String,
        secondName: String,
        keypoints: [String: SwingKeypoint]
    ) -> (x: Double, y: Double)? {
        guard
            let first = keypoints[firstName],
            let second = keypoints[secondName],
            first.score >= drawableScore,
            second.score >= drawableScore,
            var axis = normalizeDelta(dx: second.x - first.x, dy: second.y - first.y)
        else { return nil }

        if axis.x < 0 {
            axis = (x: -axis.x, y: -axis.y)
        }
        return axis
    }

    private func normalizedVector(from first: SwingKeypoint, to second: SwingKeypoint) -> (x: Double, y: Double)? {
        normalizeDelta(dx: second.x - first.x, dy: second.y - first.y)
    }

    private func normalizeDelta(dx: Double, dy: Double) -> (x: Double, y: Double)? {
        let length = hypot(dx, dy)
        guard length >= 0.01 else { return nil }
        return (x: dx / length, y: dy / length)
    }

    private func dotProduct(_ first: (x: Double, y: Double), _ second: (x: Double, y: Double)) -> Double {
        first.x * second.x + first.y * second.y
    }
}
