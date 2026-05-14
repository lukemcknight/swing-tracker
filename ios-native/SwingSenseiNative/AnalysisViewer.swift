import SwiftUI

struct AnalysisViewer: View {
    private static let minimumClubPathAnalysisVersion = 25

    @StateObject private var viewModel: SwingAnalysisViewModel
    let onClose: () -> Void

    init(
        swingID: String,
        videoURL: URL,
        analysis: SwingAnalysis,
        club: String?,
        aiAnalysis: SwingAIFeedback?,
        videoEditState: VideoEditState = .identity,
        onClubChange: @escaping (String) -> Void,
        onRequestAIFeedback: @escaping (String) async throws -> SwingAIFeedback,
        onClose: @escaping () -> Void
    ) {
        self.onClose = onClose
        _viewModel = StateObject(
            wrappedValue: SwingAnalysisViewModel(
                videoURL: videoURL,
                analysis: analysis,
                videoEditState: videoEditState
            )
        )
    }

    var body: some View {
        viewerStage
            .background(Color.black.ignoresSafeArea())
        .onAppear { viewModel.startObserving() }
        .onDisappear { viewModel.stopObserving() }
    }

    private var viewerStage: some View {
        VStack(spacing: 0) {
            viewerTopBar
            videoSurface
            analysisControlDeck
        }
        .background(.black)
    }

    private var viewerTopBar: some View {
        HStack(spacing: 18) {
            Button(action: onClose) {
                Image(systemName: "chevron.left")
                    .font(.title.weight(.medium))
                    .foregroundStyle(.white)
                    .frame(width: 50, height: 54)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Close analysis")

            Spacer()

            VStack(alignment: .trailing, spacing: 4) {
                Text("Club Path")
                    .font(.headline.weight(.black))
                    .foregroundStyle(.white)

                pathStatusBadge
            }
        }
        .padding(.horizontal, 18)
        .frame(height: 86)
        .background(.black)
    }

    private var videoSurface: some View {
        ZStack(alignment: .topTrailing) {
            TransformedPlayerView(player: viewModel.player, editState: viewModel.videoEditState)
                .background(.black)

            SkeletonOverlay(
                frame: viewModel.currentFrame,
                trailFrames: viewModel.trailFrames,
                clubPathFrames: viewModel.clubPathFrames,
                coordinateSpace: viewModel.videoCoordinateSpace,
                selectedOverlays: activeOverlays,
                analysis: viewModel.analysis
            )
                .padding(0)

            if !viewModel.isPlaying {
                Button {
                    viewModel.togglePlay()
                } label: {
                    Image(systemName: "play.fill")
                        .font(.system(size: 42, weight: .black))
                        .foregroundStyle(Theme.primary)
                        .frame(width: 82, height: 82)
                        .background(.black.opacity(0.42), in: Circle())
                }
                .buttonStyle(.plain)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
                .accessibilityLabel("Play")
            }
        }
        .aspectRatio(viewModel.videoSize.width / viewModel.videoSize.height, contentMode: .fit)
        .contentShape(Rectangle())
        .onTapGesture {
            viewModel.togglePlay()
        }
    }

    private var analysisControlDeck: some View {
        VStack(spacing: 0) {
            if showsPathStatusBanner {
                analysisQualityBanner
                    .padding(.horizontal, 18)
                    .padding(.top, 14)
                    .padding(.bottom, 10)
                    .background(Theme.surface)
            }

            phaseChips
                .padding(.horizontal, 18)
                .padding(.top, 14)
                .padding(.bottom, 10)
                .background(Theme.surface)

            HStack(spacing: 14) {
                Button {
                    viewModel.stepFrame(-1)
                } label: {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 34, weight: .medium))
                        .foregroundStyle(Theme.primary)
                        .frame(width: 46, height: 78)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Previous frame")

                TimelineRuler(
                    playheadMs: viewModel.playheadMs,
                    durationMs: viewModel.analysis.durationMs,
                    onSeek: viewModel.seek(toMs:)
                )
                .frame(height: 78)

                Button {
                    viewModel.stepFrame(1)
                } label: {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 34, weight: .medium))
                        .foregroundStyle(Theme.primary)
                        .frame(width: 46, height: 78)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Next frame")
            }
            .padding(.horizontal, 18)
            .background(Color(red: 0.18, green: 0.22, blue: 0.19))

            HStack {
                toolbarButton(
                    systemName: "speedometer",
                    title: String(format: "%.2f", viewModel.playbackRate),
                    label: "Playback speed",
                    action: cyclePlaybackRate
                )

                Spacer()

                toolbarButton(
                    systemName: viewModel.isPlaying ? "pause.fill" : "play.fill",
                    title: nil,
                    label: viewModel.isPlaying ? "Pause" : "Play",
                    action: viewModel.togglePlay
                )

                Spacer()

                toolbarButton(
                    systemName: "backward.frame",
                    title: nil,
                    label: "Previous frame",
                    action: { viewModel.stepFrame(-1) }
                )

                Spacer()

                toolbarButton(
                    systemName: "forward.frame",
                    title: nil,
                    label: "Next frame",
                    action: { viewModel.stepFrame(1) }
                )
            }
            .padding(.horizontal, 28)
            .padding(.top, 18)
            .padding(.bottom, 24)
            .background(Theme.surface)
        }
    }

    private func toolbarButton(
        systemName: String,
        title: String?,
        label: String,
        color: Color = .white,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            VStack(spacing: 5) {
                Image(systemName: systemName)
                    .font(.system(size: 29, weight: .regular))
                    .foregroundStyle(color)
                    .frame(width: 48, height: 36)

                if let title {
                    Text(title)
                        .font(.caption.weight(.bold))
                        .foregroundStyle(.white)
                        .lineLimit(1)
                }
            }
            .frame(width: 58, height: 60)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }

    private var activeOverlays: Set<AnalysisOverlay> {
        clubHeadPointCount > 0 ? [.clubPath] : []
    }

    private func cyclePlaybackRate() {
        switch viewModel.playbackRate {
        case 0.25:
            viewModel.setPlaybackRate(0.5)
        case 0.5:
            viewModel.setPlaybackRate(1.0)
        default:
            viewModel.setPlaybackRate(0.25)
        }
    }

    private var clubHeadPointCount: Int {
        viewModel.analysis.primaryClubPathSegment()?.samples.count ?? 0
    }

    private var needsReanalysisForClubData: Bool {
        (viewModel.analysis.analysisVersion ?? 0) < Self.minimumClubPathAnalysisVersion && clubHeadPointCount == 0
    }

    private var phaseChips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(viewModel.phaseMarkers) { marker in
                    Button {
                        viewModel.seek(to: marker.position)
                    } label: {
                        Text(marker.label)
                            .font(.headline.weight(.black))
                            .foregroundStyle(marker.position == viewModel.selectedPosition ? .black : .white)
                            .padding(.horizontal, 22)
                            .frame(height: 54)
                            .background(
                                marker.position == viewModel.selectedPosition ? Theme.primary : Theme.surface,
                                in: Capsule()
                            )
                            .overlay(Capsule().stroke(Theme.line))
                    }
                }
            }
            .padding(.vertical, 2)
        }
    }

    @ViewBuilder
    private var analysisQualityBanner: some View {
        if clubHeadPointCount == 0 {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.headline.weight(.black))
                    .foregroundStyle(Theme.accent)

                VStack(alignment: .leading, spacing: 4) {
                    Text(needsReanalysisForClubData ? "Reanalysis needed" : "Club path unavailable")
                        .font(.callout.weight(.black))
                        .foregroundStyle(.white)

                    Text(needsReanalysisForClubData ? "This swing was analyzed before club tracking was added." : "The club was not detected clearly enough in this clip.")
                        .font(.callout)
                        .foregroundStyle(Theme.muted)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(14)
            .background(Theme.accent.opacity(0.12), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(Theme.accent.opacity(0.34), lineWidth: 1)
            )
        } else if let quality = viewModel.analysis.analysisQuality, quality.status != "ok" {
            VStack(alignment: .leading, spacing: 6) {
                Text("Path confidence: \(pathStatusTitle)")
                    .font(.callout.weight(.black))
                    .foregroundStyle(.white)

                Text("\(cameraAngleLabel(quality.cameraAngle)) capture. \(qualityWarningCopy(quality))")
                    .font(.callout)
                    .foregroundStyle(Theme.muted)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(Theme.accent.opacity(0.12), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(Theme.accent.opacity(0.34), lineWidth: 1)
            )
        }
    }

    private func cameraAngleLabel(_ angle: String) -> String {
        AnalysisQuality.cameraAngleDisplayName(for: angle)
    }

    private func qualityWarningCopy(_ quality: AnalysisQuality) -> String {
        let warnings = Set(quality.warnings)
        if warnings.contains("club_path_uncertain") {
            return "Club tracking was unstable, so use this path as a rough visual check."
        }
        if warnings.contains("low_phase_confidence") {
            return "Some checkpoints were low confidence, so the phase markers may be approximate."
        }
        if warnings.contains("poor_framing") || warnings.contains("golfer_too_small") {
            return "Frame the full body a little larger next time for stronger tracking."
        }
        if warnings.contains("unusual_camera_angle") {
            return "Use a clearer down-the-line or face-on view for stronger tracking."
        }
        return "Review the path visually before using it for comparison."
    }

    private var showsPathStatusBanner: Bool {
        clubHeadPointCount == 0 || viewModel.analysis.analysisQuality?.status != "ok"
    }

    private var pathStatusBadge: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(pathStatusColor)
                .frame(width: 8, height: 8)

            Text(pathStatusTitle)
                .font(.caption.weight(.black))
                .foregroundStyle(.white)
        }
        .padding(.horizontal, 11)
        .frame(height: 28)
        .background(.white.opacity(0.10), in: Capsule())
        .overlay(Capsule().stroke(pathStatusColor.opacity(0.65), lineWidth: 1))
    }

    private var pathStatusTitle: String {
        guard clubHeadPointCount > 0 else { return "No Path" }
        switch viewModel.analysis.analysisQuality?.status {
        case "poor":
            return "Poor"
        case "warning":
            return "Usable"
        default:
            return "Good"
        }
    }

    private var pathStatusColor: Color {
        guard clubHeadPointCount > 0 else { return .red.opacity(0.92) }
        switch viewModel.analysis.analysisQuality?.status {
        case "poor":
            return .red.opacity(0.92)
        case "warning":
            return Theme.accent
        default:
            return Theme.primary
        }
    }
}

enum AnalysisOverlay: String, CaseIterable, Identifiable {
    case guides
    case bodyBox
    case headTracking
    case spineTracking
    case clubPath
    case wristPath
    case swingPlane
    case joints

    var id: String { rawValue }

    var title: String {
        switch self {
        case .guides:
            "Guides"
        case .bodyBox:
            "Body Box"
        case .headTracking:
            "Head Tracking"
        case .spineTracking:
            "Spine Tracking"
        case .clubPath:
            "Club Path"
        case .wristPath:
            "Wrist Path"
        case .swingPlane:
            "Swing Plane"
        case .joints:
            "Joints"
        }
    }
}

private struct TimelineRuler: View {
    let playheadMs: Double
    let durationMs: Int
    let onSeek: (Double) -> Void

    private var progress: CGFloat {
        guard durationMs > 0 else { return 0 }
        return CGFloat(min(max(playheadMs / Double(durationMs), 0), 1))
    }

    var body: some View {
        GeometryReader { proxy in
            let width = max(1, proxy.size.width)
            let playheadX = width * progress

            ZStack(alignment: .bottomLeading) {
                Canvas { context, size in
                    let baselineY = size.height * 0.42
                    let tickCount = 26

                    var baseline = Path()
                    baseline.move(to: CGPoint(x: 0, y: baselineY))
                    baseline.addLine(to: CGPoint(x: size.width, y: baselineY))
                    context.stroke(baseline, with: .color(.white.opacity(0.22)), lineWidth: 1)

                    for index in 0...tickCount {
                        let x = size.width * CGFloat(index) / CGFloat(tickCount)
                        let isMajor = index % 5 == 0
                        let tickHeight: CGFloat = isMajor ? 38 : 15
                        var tick = Path()
                        tick.move(to: CGPoint(x: x, y: baselineY))
                        tick.addLine(to: CGPoint(x: x, y: baselineY + tickHeight))
                        context.stroke(
                            tick,
                            with: .color(.white.opacity(isMajor ? 0.64 : 0.42)),
                            lineWidth: isMajor ? 2 : 1.4
                        )
                    }

                    var playhead = Path()
                    playhead.move(to: CGPoint(x: playheadX, y: 0))
                    playhead.addLine(to: CGPoint(x: playheadX, y: size.height))
                    context.stroke(playhead, with: .color(.red.opacity(0.9)), lineWidth: 3)
                }

                Text(Self.formatClock(playheadMs))
                    .font(.system(.title3, design: .monospaced).weight(.bold))
                    .foregroundStyle(.white.opacity(0.92))
                    .padding(.horizontal, 8)
                    .background(Color(red: 0.18, green: 0.22, blue: 0.19))
                    .position(x: min(max(playheadX, 42), width - 42), y: 61)

                Text(Self.formatClock(Double(durationMs)))
                    .font(.system(.headline, design: .monospaced).weight(.bold))
                    .foregroundStyle(.white.opacity(0.76))
                    .position(x: width - 28, y: 62)
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        seek(at: value.location.x, width: width)
                    }
            )
        }
    }

    private func seek(at x: CGFloat, width: CGFloat) {
        let nextProgress = min(max(x / max(1, width), 0), 1)
        onSeek(Double(nextProgress) * Double(max(1, durationMs)))
    }

    private static func formatClock(_ milliseconds: Double) -> String {
        let seconds = max(0, milliseconds / 1000)
        let wholeSeconds = Int(seconds)
        let tenths = Int((seconds - Double(wholeSeconds)) * 10)
        return "\(wholeSeconds):\(String(format: "%01d", tenths))"
    }
}
