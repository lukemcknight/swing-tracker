import AVFoundation
import Foundation
import PhotosUI
import SwiftUI

struct SwingSelection: Identifiable, Equatable {
    let id: String
}

struct AppMessage: Identifiable, Equatable {
    let id = UUID()
    let text: String
}

private struct PendingRoundShot: Equatable {
    let roundID: String
    let swingID: String
    let ballMode: ArcadeBallMode
}

@MainActor
final class AppViewModel: ObservableObject {
    @Published private(set) var swings: [SwingRecord] = []
    @Published private(set) var rounds: [RoundRecord] = []
    @Published var activeRoundID: String?
    @Published var trimSelection: SwingSelection?
    @Published var viewerSelection: SwingSelection?
    @Published var processingMessage: String?
    @Published var appMessage: AppMessage?
    @Published private(set) var pendingRoundShotSwingID: String?

    private let repository: SwingRepository
    private let roundRepository: RoundRepository
    private let shotEngine: ArcadeShotEngine
    private var pendingRoundShot: PendingRoundShot? {
        didSet { pendingRoundShotSwingID = pendingRoundShot?.swingID }
    }

    init(
        repository: SwingRepository = SwingRepository(),
        roundRepository: RoundRepository = RoundRepository(),
        shotEngine: ArcadeShotEngine = ArcadeShotEngine()
    ) {
        self.repository = repository
        self.roundRepository = roundRepository
        self.shotEngine = shotEngine
    }

    func loadData() {
        loadSwings()
        loadRounds()
    }

    func loadSwings() {
        do {
            swings = try repository.load()
        } catch {
            appMessage = AppMessage(text: "Could not load swing history: \(error.localizedDescription)")
        }
    }

    func loadRounds() {
        do {
            rounds = try roundRepository.load()
            if activeRoundID == nil {
                activeRoundID = rounds.first(where: { $0.status == .inProgress })?.id ?? rounds.first?.id
            }
        } catch {
            appMessage = AppMessage(text: "Could not load rounds: \(error.localizedDescription)")
        }
    }

    func swing(id: String) -> SwingRecord? {
        swings.first { $0.id == id }
    }

    var activeRound: RoundRecord? {
        guard let activeRoundID else { return rounds.first(where: { $0.status == .inProgress }) }
        return rounds.first { $0.id == activeRoundID }
    }

    var activeHole: HoleState? {
        activeRound?.currentHole
    }

    var recentCompletedRounds: [RoundRecord] {
        rounds.filter { $0.status == .complete }
    }

    func startStarterRound() {
        let round = RoundRecord.starter()
        do {
            try save(round)
            activeRoundID = round.id
            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        } catch {
            appMessage = AppMessage(text: "Could not start round: \(error.localizedDescription)")
        }
    }

    func continueRound(_ round: RoundRecord) {
        activeRoundID = round.id
    }

    func undoLastRoundShot() {
        guard var round = activeRound else { return }
        guard round.undoLastShot() else { return }
        do {
            try save(round)
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
        } catch {
            appMessage = AppMessage(text: "Could not undo shot: \(error.localizedDescription)")
        }
    }

    func restartActiveRound() {
        guard var round = activeRound else { return }
        round.restartHoles()
        do {
            try save(round)
            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        } catch {
            appMessage = AppMessage(text: "Could not restart round: \(error.localizedDescription)")
        }
    }

    func abandonActiveRound() {
        guard var round = activeRound else { return }
        round.abandon()
        do {
            try save(round)
            activeRoundID = nil
            pendingRoundShot = nil
            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        } catch {
            appMessage = AppMessage(text: "Could not abandon round: \(error.localizedDescription)")
        }
    }

    func retryPendingRoundShot(baseURLString: String) async {
        guard let pendingRoundShot else { return }
        await retryAnalysis(swingID: pendingRoundShot.swingID, baseURLString: baseURLString)
    }

    func discardPendingRoundShot() {
        pendingRoundShot = nil
    }

    func importPickedVideo(_ item: PhotosPickerItem?) async {
        guard let item else { return }

        do {
            processingMessage = "Importing video"
            guard let pickedVideo = try await item.loadTransferable(type: PickedVideo.self) else {
                appMessage = AppMessage(text: "Could not load the selected video.")
                processingMessage = nil
                return
            }

            let swing = try await persistNewSwing(videoURL: pickedVideo.url, source: .uploaded)
            trimSelection = SwingSelection(id: swing.id)
        } catch {
            appMessage = AppMessage(text: error.localizedDescription)
        }

        processingMessage = nil
    }

    func finishRecordedVideo(_ videoURL: URL) async {
        do {
            processingMessage = "Saving recording"
            let swing = try await persistNewSwing(videoURL: videoURL, source: .recorded)
            trimSelection = SwingSelection(id: swing.id)
        } catch {
            appMessage = AppMessage(text: error.localizedDescription)
        }

        processingMessage = nil
    }

    func finishRecordedRoundShotVideo(
        _ videoURL: URL,
        club: String,
        ballMode: ArcadeBallMode
    ) async {
        guard let round = activeRound, round.status == .inProgress else {
            appMessage = AppMessage(text: "Start or continue a round before recording a shot.")
            return
        }

        do {
            processingMessage = "Saving shot"
            let swing = try await persistNewSwing(
                videoURL: videoURL,
                source: .recorded,
                initialClub: club
            )
            pendingRoundShot = PendingRoundShot(
                roundID: round.id,
                swingID: swing.id,
                ballMode: ballMode
            )
            trimSelection = SwingSelection(id: swing.id)
        } catch {
            appMessage = AppMessage(text: error.localizedDescription)
        }

        processingMessage = nil
    }

    func saveTrimAndAnalyze(
        swingID: String,
        club: String,
        startMs: Int,
        endMs: Int,
        durationMs: Int,
        videoEditState: VideoEditState? = nil,
        baseURLString: String
    ) async {
        guard var swing = swing(id: swingID) else { return }

        do {
            let trim = TrimWindow.validated(startMs: Double(startMs), endMs: Double(endMs), durationMs: durationMs)
            let editState = videoEditState?.normalized
            swing.durationMs = durationMs
            swing.trimStartMs = trim.startMs
            swing.trimEndMs = trim.endMs
            swing.videoEditState = editState?.isIdentity == false ? editState : nil
            swing.club = club
            swing.aiAnalysis = nil
            swing.status = .uploading
            swing.analysisError = nil
            try save(swing)

            trimSelection = nil
            processingMessage = "Analyzing swing"

            let client = try AnalysisAPIClient(baseURLString: baseURLString)
            swing.status = .analyzing
            try save(swing)

            let backendAnalysis = try await client.analyzeSwing(
                videoURL: swing.videoURL,
                swingID: swing.id,
                club: club,
                durationMs: durationMs,
                trimStartMs: trim.startMs,
                trimEndMs: trim.endMs,
                videoEditState: swing.videoEditState
            )

            processingMessage = "Detecting club path"
            let analysis: SwingAnalysis
            do {
                let detector = try ClubHeadDetector()
                let service = ClubHeadDetectionService(detector: detector)
                analysis = try await service.populateClubHead(
                    into: backendAnalysis,
                    videoURL: swing.videoURL
                )
                print("[AppViewModel] local club path detection succeeded")
            } catch {
                print("[AppViewModel] local club path detection failed: \(error)")
                analysis = backendAnalysis
            }

            swing.analysis = analysis
            swing.senseiScore = analysis.senseiScore
            swing.status = .complete
            swing.analysisError = nil
            try save(swing)

            if let pendingRoundShot, pendingRoundShot.swingID == swing.id {
                self.pendingRoundShot = nil
                do {
                    try applyRoundShot(
                        roundID: pendingRoundShot.roundID,
                        swing: swing,
                        club: club,
                        ballMode: pendingRoundShot.ballMode,
                        analysis: analysis
                    )
                    processingMessage = nil
                    UINotificationFeedbackGenerator().notificationOccurred(.success)
                    return
                } catch {
                    appMessage = AppMessage(text: "Swing analyzed, but the shot could not be added to the round: \(error.localizedDescription)")
                }
            }

            processingMessage = nil
            viewerSelection = SwingSelection(id: swing.id)
            UINotificationFeedbackGenerator().notificationOccurred(.success)
        } catch {
            processingMessage = nil
            if var failed = self.swing(id: swingID) {
                failed.status = .failed
                failed.analysisError = error.localizedDescription
                try? save(failed)
            }
            UINotificationFeedbackGenerator().notificationOccurred(.error)
            appMessage = AppMessage(text: error.localizedDescription)
        }
    }

    func retryAnalysis(swingID: String, baseURLString: String) async {
        guard let swing = swing(id: swingID) else { return }
        let club = swing.club ?? GolfClub.other.rawValue
        let durationMs = swing.durationMs ?? 0
        let startMs = swing.trimStartMs ?? 0
        let endMs = swing.trimEndMs ?? max(durationMs, 1)
        await saveTrimAndAnalyze(
            swingID: swingID,
            club: club,
            startMs: startMs,
            endMs: endMs,
            durationMs: durationMs,
            videoEditState: swing.videoEditState,
            baseURLString: baseURLString
        )
    }

    func updateClub(swingID: String, club: String) {
        guard var swing = swing(id: swingID), swing.club != club else { return }
        swing.club = club
        do {
            try save(swing)
        } catch {
            appMessage = AppMessage(text: "Could not save club selection: \(error.localizedDescription)")
        }
    }

    func requestAIFeedback(swingID: String, club: String, baseURLString: String) async throws -> SwingAIFeedback {
        guard var swing = swing(id: swingID), let analysis = swing.analysis else {
            throw AnalysisAPIError.invalidResponse
        }

        let client = try AnalysisAPIClient(baseURLString: baseURLString)
        let feedback = try await client.requestAIFeedback(
            videoURL: swing.videoURL,
            swingID: swing.id,
            club: club,
            analysis: analysis,
            durationMs: swing.durationMs,
            trimStartMs: swing.trimStartMs,
            trimEndMs: swing.trimEndMs,
            videoEditState: swing.videoEditState
        )

        swing.club = club
        swing.aiAnalysis = feedback
        try save(swing)
        UINotificationFeedbackGenerator().notificationOccurred(.success)
        return feedback
    }

    func open(_ swing: SwingRecord) {
        if swing.analysis != nil, swing.status == .complete || swing.status == .failed {
            viewerSelection = SwingSelection(id: swing.id)
            return
        }

        switch swing.status {
        case .uploading, .analyzing:
            appMessage = AppMessage(text: "This swing is already being analyzed.")
        default:
            trimSelection = SwingSelection(id: swing.id)
        }
    }

    func cancelTrimSelection() {
        if
            let trimSelection,
            pendingRoundShot?.swingID == trimSelection.id
        {
            pendingRoundShot = nil
        }
        trimSelection = nil
    }

    private func persistNewSwing(
        videoURL: URL,
        source: SwingSource,
        initialClub: String? = nil
    ) async throws -> SwingRecord {
        let durationMs = try await loadDurationMs(videoURL: videoURL)
        let orientation = try await loadOrientation(videoURL: videoURL)
        var swing = try repository.copyVideo(
            from: videoURL,
            source: source,
            durationMs: durationMs,
            orientation: orientation
        )
        swing.club = initialClub
        try save(swing)
        UINotificationFeedbackGenerator().notificationOccurred(.success)
        return swing
    }

    private func save(_ swing: SwingRecord) throws {
        swings = repository.upserting(swing, into: swings)
        try repository.save(swings)
    }

    private func save(_ round: RoundRecord) throws {
        rounds = roundRepository.upserting(round, into: rounds)
        try roundRepository.save(rounds)
    }

    private func applyRoundShot(
        roundID: String,
        swing: SwingRecord,
        club: String,
        ballMode: ArcadeBallMode,
        analysis: SwingAnalysis
    ) throws {
        guard var round = rounds.first(where: { $0.id == roundID }) else {
            throw RoundRepositoryError.roundNotFound(roundID)
        }
        guard let hole = round.currentHole else {
            return
        }

        let result = shotEngine.resolveShot(
            hole: hole,
            club: club,
            ballMode: ballMode,
            swingID: swing.id,
            shotEstimate: analysis.shotEstimate
        )
        round.replaceHole(result.updatedHole)
        try save(round)
        activeRoundID = round.id
    }

    private func loadDurationMs(videoURL: URL) async throws -> Int {
        let asset = AVURLAsset(url: videoURL)
        let duration = try await asset.load(.duration)
        let seconds = CMTimeGetSeconds(duration)
        guard seconds.isFinite && seconds > 0 else { return 0 }
        return Int((seconds * 1000).rounded())
    }

    private func loadOrientation(videoURL: URL) async throws -> SwingOrientation {
        let asset = AVURLAsset(url: videoURL)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            return .unknown
        }

        let size = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let transformedSize = size.applying(transform)
        let width = abs(transformedSize.width)
        let height = abs(transformedSize.height)

        if width > height {
            return .landscape
        }
        if height > width {
            return .portrait
        }
        return .unknown
    }
}
