import AVFoundation
import Foundation

@MainActor
final class SwingAnalysisViewModel: ObservableObject {
    let analysis: SwingAnalysis
    let player: AVPlayer
    let videoEditState: VideoEditState

    @Published private(set) var playheadMs: Double = 0
    @Published private(set) var currentFrameIndex: Int = 0
    @Published private(set) var videoCoordinateSpace: VideoCoordinateSpace
    @Published var selectedPosition: SwingPosition = .takeaway
    @Published private(set) var isPlaying = false
    @Published private(set) var playbackRate: Float = 1

    private let videoURL: URL
    private var timeObserver: Any?
    private var seekGeneration = 0
    private var didLoadVideoCoordinateSpace = false

    init(videoURL: URL, analysis: SwingAnalysis, videoEditState: VideoEditState = .identity) {
        self.analysis = analysis
        self.videoURL = videoURL
        self.videoEditState = videoEditState.normalized
        self.player = AVPlayer(url: videoURL)
        self.videoCoordinateSpace = .identity(size: Self.analysisVideoSize(analysis))
        self.currentFrameIndex = analysis.nearestFrameIndex(to: 0) ?? 0
        self.selectedPosition = analysis.closestPosition(to: 0)
    }

    deinit {
        if let timeObserver {
            player.removeTimeObserver(timeObserver)
        }
    }

    var currentFrame: SwingAnalysisFrame? {
        guard analysis.frames.indices.contains(currentFrameIndex) else { return nil }
        return analysis.frames[currentFrameIndex]
    }

    var trailFrames: [SwingAnalysisFrame] {
        guard !analysis.frames.isEmpty else { return [] }
        let start = max(0, currentFrameIndex - 12)
        return Array(analysis.frames[start...currentFrameIndex])
    }

    var clubPathFrames: [SwingAnalysisFrame] {
        guard !analysis.frames.isEmpty else { return [] }
        let lastIndex = analysis.frames.count - 1
        let playheadEnd = min(max(currentFrameIndex, 0), lastIndex)
        let finishEnd = analysis.frameIndex(for: .finish) ?? lastIndex
        let end = min(playheadEnd, finishEnd)
        return Array(analysis.frames[0...end])
    }

    var phaseMarkers: [PhaseMarker] {
        analysis.phaseMarkers()
    }

    var selectedMarker: PhaseMarker? {
        analysis.marker(for: selectedPosition)
    }

    var videoSize: CGSize {
        videoCoordinateSpace.displaySize
    }

    func startObserving() {
        guard timeObserver == nil else { return }
        timeObserver = player.addPeriodicTimeObserver(
            forInterval: CMTime(seconds: 0.03, preferredTimescale: 600),
            queue: .main
        ) { [weak self] time in
            Task { @MainActor in
                self?.sync(to: CMTimeGetSeconds(time) * 1000)
            }
        }
        Task { await loadVideoCoordinateSpace() }
    }

    func stopObserving() {
        if let timeObserver {
            player.removeTimeObserver(timeObserver)
            self.timeObserver = nil
        }
        player.pause()
        isPlaying = false
    }

    func togglePlay() {
        if isPlaying {
            player.pause()
            isPlaying = false
            return
        }

        player.rate = playbackRate
        isPlaying = true
    }

    func setPlaybackRate(_ rate: Float) {
        playbackRate = rate
        if isPlaying {
            player.rate = rate
        }
    }

    func seek(to position: SwingPosition) {
        guard let marker = analysis.marker(for: position) else { return }
        selectedPosition = position
        seek(toMs: Double(marker.timeMs))
    }

    func stepFrame(_ delta: Int) {
        guard !analysis.frames.isEmpty else { return }
        let nextIndex = min(max(currentFrameIndex + delta, 0), analysis.frames.count - 1)
        let nextTimeMs = analysis.frames[nextIndex].t
        selectedPosition = analysis.closestPosition(to: Double(nextTimeMs))
        seek(toMs: Double(nextTimeMs))
    }

    func seek(toMs timeMs: Double) {
        let boundedMs = min(max(timeMs, 0), Double(max(1, analysis.durationMs)))
        let time = CMTime(seconds: boundedMs / 1000, preferredTimescale: 600)
        seekGeneration += 1
        let generation = seekGeneration

        player.seek(to: time, toleranceBefore: .zero, toleranceAfter: .zero) { [weak self] finished in
            guard finished, let self else { return }

            Task { @MainActor in
                guard generation == self.seekGeneration else { return }
                self.sync(to: CMTimeGetSeconds(self.player.currentTime()) * 1000)
            }
        }
    }

    private func sync(to timeMs: Double) {
        playheadMs = min(max(timeMs, 0), Double(max(1, analysis.durationMs)))
        if let frameIndex = analysis.nearestFrameIndex(to: playheadMs) {
            currentFrameIndex = frameIndex
        }
        selectedPosition = analysis.closestPosition(to: playheadMs)
    }

    private func loadVideoCoordinateSpace() async {
        guard !didLoadVideoCoordinateSpace else { return }
        didLoadVideoCoordinateSpace = true

        guard videoEditState.isIdentity else {
            videoCoordinateSpace = .identity(size: Self.analysisVideoSize(analysis))
            return
        }

        do {
            let asset = AVURLAsset(url: videoURL)
            guard let track = try await asset.loadTracks(withMediaType: .video).first else { return }
            let naturalSize = try await track.load(.naturalSize)
            let preferredTransform = try await track.load(.preferredTransform)
            videoCoordinateSpace = .displayingAsset(
                analysisSize: Self.analysisVideoSize(analysis),
                naturalSize: naturalSize,
                preferredTransform: preferredTransform
            )
        } catch {
            videoCoordinateSpace = .identity(size: Self.analysisVideoSize(analysis))
        }
    }

    private static func analysisVideoSize(_ analysis: SwingAnalysis) -> CGSize {
        CGSize(
            width: max(1, analysis.videoWidth ?? 1080),
            height: max(1, analysis.videoHeight ?? 1920)
        )
    }
}
