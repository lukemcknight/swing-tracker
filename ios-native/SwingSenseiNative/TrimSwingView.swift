import AVFoundation
import SwiftUI
import UIKit

struct TrimSwingView: View {
    let swing: SwingRecord
    let onCancel: () -> Void
    let onAnalyze: (Int, Int, Int, VideoEditState, String) async -> Void

    @StateObject private var viewModel: TrimSwingViewModel
    @State private var isAnalyzing = false

    init(
        swing: SwingRecord,
        onCancel: @escaping () -> Void,
        onAnalyze: @escaping (Int, Int, Int, VideoEditState, String) async -> Void
    ) {
        self.swing = swing
        self.onCancel = onCancel
        self.onAnalyze = onAnalyze
        _viewModel = StateObject(wrappedValue: TrimSwingViewModel(swing: swing))
    }

    var body: some View {
        VStack(spacing: 0) {
            header

            ScrollView {
                VStack(spacing: 22) {
                    videoPreview
                    playRow
                    timeline
                    editControls
                    analyzeButton
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 28)
            }
        }
        .background(Theme.background.ignoresSafeArea())
        .task {
            await viewModel.load()
        }
        .onDisappear {
            viewModel.stop()
        }
    }

    private var header: some View {
        HStack {
            Button(action: onCancel) {
                Image(systemName: "xmark")
                    .font(.headline.weight(.black))
                    .foregroundStyle(.white)
                    .frame(width: 46, height: 46)
                    .background(Theme.surface, in: Circle())
            }
            .disabled(isAnalyzing)

            Spacer()

            Text("Edit Swing")
                .font(.headline.weight(.black))
                .foregroundStyle(.white)

            Spacer()

            Color.clear.frame(width: 46, height: 46)
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 12)
    }

    private var videoPreview: some View {
        ZStack {
            TransformedPlayerView(player: viewModel.player, editState: viewModel.videoEditState)
                .background(.black)
        }
        .aspectRatio(viewModel.displayAspectRatio, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 22, style: .continuous).stroke(Theme.line))
        .padding(.top, 4)
    }

    private var playRow: some View {
        HStack(spacing: 14) {
            Text("\(formatClock(viewModel.playheadMs)) / \(formatClock(viewModel.durationMs))")
                .font(.system(.callout, design: .monospaced).weight(.bold))
                .foregroundStyle(Theme.muted)

            Spacer()

            Button {
                viewModel.togglePlayback()
            } label: {
                Image(systemName: viewModel.isPlaying ? "pause.fill" : "play.fill")
                    .font(.headline.weight(.black))
                    .foregroundStyle(.black)
                    .frame(width: 54, height: 54)
                    .background(Theme.primary, in: Circle())
            }
        }
    }

    private var timeline: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Select the swing window")
                    .font(.caption.weight(.black))
                    .foregroundStyle(Theme.primary)
                    .textCase(.uppercase)
                    .tracking(1.2)

                Spacer()

                Text("\(formatClock(viewModel.trimStartMs)) - \(formatClock(viewModel.trimEndMs))")
                    .font(.system(.caption, design: .monospaced).weight(.bold))
                    .foregroundStyle(Theme.muted)
            }

            TrimTimeline(
                thumbnails: viewModel.thumbnails,
                durationMs: viewModel.durationMs,
                playheadMs: viewModel.playheadMs,
                trimStartMs: viewModel.trimStartMs,
                trimEndMs: viewModel.trimEndMs,
                onSeek: viewModel.seek,
                onTrimChange: viewModel.updateTrim(startMs:endMs:)
            )
            .frame(height: 96)
        }
        .padding(16)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Theme.line))
    }

    private var editControls: some View {
        VStack(spacing: 12) {
            HStack(spacing: 28) {
                EditorToolButton(
                    title: "Flip",
                    systemImage: "arrow.left.and.right",
                    isActive: viewModel.videoEditState.isMirroredHorizontally,
                    isDisabled: isAnalyzing,
                    action: viewModel.toggleHorizontalMirror
                )

                EditorToolButton(
                    title: "Rotate",
                    systemImage: "rotate.right",
                    isActive: viewModel.videoEditState.rotationDegrees != 0,
                    isDisabled: isAnalyzing,
                    action: viewModel.rotateClockwise
                )

                if viewModel.hasVideoEdits {
                    EditorToolButton(
                        title: "Reset",
                        systemImage: "arrow.counterclockwise",
                        isActive: true,
                        isDisabled: isAnalyzing,
                        action: viewModel.resetVideoEdits
                    )
                    .transition(.scale.combined(with: .opacity))
                }
            }

            if viewModel.hasVideoEdits {
                Text(viewModel.editSummary)
                    .font(.system(.caption, design: .monospaced).weight(.bold))
                    .foregroundStyle(Theme.muted)
                    .transition(.opacity)
            }
        }
        .animation(.spring(response: 0.24, dampingFraction: 0.82), value: viewModel.videoEditState)
    }

    private var analyzeButton: some View {
        Button {
            Task {
                isAnalyzing = true
                viewModel.pause()
                await onAnalyze(
                    viewModel.trimStartMs,
                    viewModel.trimEndMs,
                    viewModel.durationMs,
                    viewModel.videoEditState,
                    GolfClub.other.rawValue
                )
                isAnalyzing = false
            }
        } label: {
            HStack(spacing: 10) {
                if isAnalyzing {
                    ProgressView()
                        .tint(.black)
                } else {
                    Image(systemName: "point.3.connected.trianglepath.dotted")
                        .font(.headline.weight(.black))
                }

                Text(isAnalyzing ? "Analyzing" : "Analyze Club Path")
                    .font(.title3.weight(.black))
            }
            .foregroundStyle(.black)
            .frame(maxWidth: .infinity)
            .frame(height: 58)
            .background(canAnalyze ? Theme.primary : Theme.elevated, in: Capsule())
        }
        .disabled(!canAnalyze || isAnalyzing)
    }

    private var canAnalyze: Bool {
        viewModel.canAnalyze
    }

    private func formatClock(_ milliseconds: Int) -> String {
        let totalSeconds = max(0, milliseconds) / 1000
        return "\(String(format: "%02d", totalSeconds / 60)):\(String(format: "%02d", totalSeconds % 60))"
    }
}

@MainActor
final class TrimSwingViewModel: ObservableObject {
    let player: AVPlayer

    @Published private(set) var thumbnails: [TrimThumbnail] = []
    @Published private(set) var durationMs: Int
    @Published private(set) var playheadMs: Int
    @Published private(set) var isPlaying = false
    @Published var trimStartMs: Int
    @Published var trimEndMs: Int
    @Published var videoEditState: VideoEditState
    @Published private(set) var videoAspectRatio: CGFloat = 9.0 / 16.0

    private let swing: SwingRecord
    private var timeObserver: Any?

    init(swing: SwingRecord) {
        self.swing = swing
        self.player = AVPlayer(url: swing.videoURL)
        let initialDuration = max(0, swing.durationMs ?? 0)
        let initialStart = swing.trimStartMs ?? 0
        let initialEnd = swing.trimEndMs ?? max(initialDuration, 1)
        self.durationMs = initialDuration
        self.trimStartMs = initialStart
        self.trimEndMs = initialEnd
        self.videoEditState = swing.videoEditState ?? .identity
        self.playheadMs = initialStart
    }

    deinit {
        if let timeObserver {
            player.removeTimeObserver(timeObserver)
        }
    }

    var canAnalyze: Bool {
        TrimWindow.canContinue(
            startMs: Double(trimStartMs),
            endMs: Double(trimEndMs),
            durationMs: durationMs
        )
    }

    var displayAspectRatio: CGFloat {
        videoEditState.rotatesDimensions ? 1 / max(videoAspectRatio, 0.01) : videoAspectRatio
    }

    var hasVideoEdits: Bool {
        !videoEditState.isIdentity
    }

    var editSummary: String {
        var parts: [String] = []
        if videoEditState.rotationDegrees != 0 {
            parts.append("Rotated \(videoEditState.rotationDegrees) deg")
        }
        if videoEditState.isMirroredHorizontally {
            parts.append("Mirrored")
        }
        return parts.joined(separator: " / ")
    }

    func load() async {
        do {
            let asset = AVURLAsset(url: swing.videoURL)
            let duration = try await asset.load(.duration)
            let seconds = CMTimeGetSeconds(duration)
            let loadedDurationMs = seconds.isFinite && seconds > 0 ? Int((seconds * 1000).rounded()) : durationMs
            durationMs = max(durationMs, loadedDurationMs)

            if let track = try await asset.loadTracks(withMediaType: .video).first {
                let size = try await track.load(.naturalSize)
                let transform = try await track.load(.preferredTransform)
                let transformed = size.applying(transform)
                let width = max(1, abs(transformed.width))
                let height = max(1, abs(transformed.height))
                videoAspectRatio = width / height
            }

            let trim = TrimWindow.validated(
                startMs: Double(trimStartMs),
                endMs: Double(trimEndMs),
                durationMs: durationMs
            )
            trimStartMs = trim.startMs
            trimEndMs = trim.endMs
            playheadMs = trim.startMs
            seek(trim.startMs)
            startObserving()
            thumbnails = await Self.generateThumbnails(url: swing.videoURL, durationMs: durationMs)
        } catch {
            startObserving()
        }
    }

    func togglePlayback() {
        if isPlaying {
            pause()
            return
        }

        if playheadMs < trimStartMs || playheadMs >= trimEndMs {
            seek(trimStartMs)
        }

        UIImpactFeedbackGenerator(style: .light).impactOccurred()
        player.play()
        isPlaying = true
    }

    func pause() {
        player.pause()
        isPlaying = false
    }

    func stop() {
        pause()
        if let timeObserver {
            player.removeTimeObserver(timeObserver)
            self.timeObserver = nil
        }
    }

    func seek(_ milliseconds: Int) {
        let bounded = max(0, min(milliseconds, max(1, durationMs)))
        playheadMs = bounded
        let time = CMTime(seconds: Double(bounded) / 1000, preferredTimescale: 600)
        player.seek(to: time, toleranceBefore: .zero, toleranceAfter: .zero)
    }

    func updateTrim(startMs: Int, endMs: Int) {
        let trim = TrimWindow.validated(
            startMs: Double(startMs),
            endMs: Double(endMs),
            durationMs: durationMs
        )
        trimStartMs = trim.startMs
        trimEndMs = trim.endMs
        seek(max(trim.startMs, min(playheadMs, trim.endMs)))
    }

    func toggleHorizontalMirror() {
        videoEditState = videoEditState.togglingHorizontalMirror()
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }

    func rotateClockwise() {
        videoEditState = videoEditState.rotatedClockwise()
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }

    func resetVideoEdits() {
        videoEditState = .identity
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
    }

    private func startObserving() {
        guard timeObserver == nil else { return }
        timeObserver = player.addPeriodicTimeObserver(
            forInterval: CMTime(seconds: 0.05, preferredTimescale: 600),
            queue: .main
        ) { [weak self] time in
            Task { @MainActor in
                self?.sync(to: CMTimeGetSeconds(time) * 1000)
            }
        }
    }

    private func sync(to timeMs: Double) {
        let current = max(0, min(Int(timeMs.rounded()), max(1, durationMs)))
        if isPlaying, current >= trimEndMs {
            seek(trimStartMs)
            player.play()
            return
        }
        playheadMs = current
    }

    private static func generateThumbnails(url: URL, durationMs: Int) async -> [TrimThumbnail] {
        await Task.detached(priority: .utility) {
            let asset = AVURLAsset(url: url)
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            generator.maximumSize = CGSize(width: 180, height: 180)

            let durationSeconds = max(0.1, Double(durationMs) / 1000)
            return (0..<12).compactMap { index in
                let progress = Double(index) / 11
                let seconds = min(durationSeconds - 0.04, max(0, durationSeconds * progress))
                do {
                    let image = try generator.copyCGImage(
                        at: CMTime(seconds: seconds, preferredTimescale: 600),
                        actualTime: nil
                    )
                    return TrimThumbnail(image: UIImage(cgImage: image))
                } catch {
                    return nil
                }
            }
        }.value
    }
}

struct TrimThumbnail: Identifiable {
    let id = UUID()
    let image: UIImage
}

private struct EditorToolButton: View {
    let title: String
    let systemImage: String
    let isActive: Bool
    let isDisabled: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 8) {
                Image(systemName: systemImage)
                    .font(.title2.weight(.black))
                    .foregroundStyle(isActive ? .black : .white)
                    .frame(width: 58, height: 58)
                    .background(isActive ? Theme.primary : Theme.surface, in: Circle())
                    .overlay(Circle().stroke(isActive ? Theme.primary.opacity(0.85) : Theme.line, lineWidth: 2))

                Text(title)
                    .font(.caption.weight(.black))
                    .foregroundStyle(isActive ? Theme.primary : Theme.muted)
            }
        }
        .buttonStyle(.plain)
        .disabled(isDisabled)
        .opacity(isDisabled ? 0.48 : 1)
        .accessibilityLabel(title)
    }
}

private struct TrimTimeline: View {
    let thumbnails: [TrimThumbnail]
    let durationMs: Int
    let playheadMs: Int
    let trimStartMs: Int
    let trimEndMs: Int
    let onSeek: (Int) -> Void
    let onTrimChange: (Int, Int) -> Void

    var body: some View {
        GeometryReader { geometry in
            let width = max(1, geometry.size.width)
            let startX = xPosition(for: trimStartMs, width: width)
            let endX = xPosition(for: trimEndMs, width: width)
            let playheadX = xPosition(for: playheadMs, width: width)

            ZStack(alignment: .leading) {
                thumbnailStrip
                    .frame(width: width, height: geometry.size.height)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))

                Rectangle()
                    .fill(.black.opacity(0.58))
                    .frame(width: startX)

                Rectangle()
                    .fill(.black.opacity(0.58))
                    .frame(width: max(0, width - endX))
                    .offset(x: endX)

                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(Theme.primary, lineWidth: 3)
                    .frame(width: max(10, endX - startX), height: geometry.size.height)
                    .offset(x: startX)

                Capsule()
                    .fill(.white)
                    .frame(width: 3, height: geometry.size.height + 10)
                    .offset(x: playheadX - 1.5, y: -5)

                trimHandle
                    .offset(x: max(0, startX - 12))
                    .gesture(
                        DragGesture()
                            .onChanged { value in
                                let nextStart = milliseconds(for: value.location.x, width: width)
                                onTrimChange(nextStart, trimEndMs)
                            }
                    )

                trimHandle
                    .offset(x: min(width - 24, endX - 12))
                    .gesture(
                        DragGesture()
                            .onChanged { value in
                                let nextEnd = milliseconds(for: value.location.x, width: width)
                                onTrimChange(trimStartMs, nextEnd)
                            }
                    )
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        onSeek(milliseconds(for: value.location.x, width: width))
                    }
            )
        }
    }

    private var thumbnailStrip: some View {
        HStack(spacing: 0) {
            if thumbnails.isEmpty {
                Rectangle()
                    .fill(Theme.elevated)
                    .overlay {
                        Image(systemName: "film.fill")
                            .foregroundStyle(Theme.primary)
                    }
            } else {
                ForEach(thumbnails) { thumbnail in
                    Image(uiImage: thumbnail.image)
                        .resizable()
                        .scaledToFill()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .clipped()
                }
            }
        }
    }

    private var trimHandle: some View {
        RoundedRectangle(cornerRadius: 5, style: .continuous)
            .fill(Theme.primary)
            .frame(width: 24, height: 96)
            .overlay {
                Capsule()
                    .fill(.black.opacity(0.42))
                    .frame(width: 4, height: 44)
            }
    }

    private func xPosition(for milliseconds: Int, width: CGFloat) -> CGFloat {
        CGFloat(max(0, min(Double(milliseconds) / Double(max(1, durationMs)), 1))) * width
    }

    private func milliseconds(for x: CGFloat, width: CGFloat) -> Int {
        Int((max(0, min(x / max(1, width), 1)) * CGFloat(max(1, durationMs))).rounded())
    }
}
