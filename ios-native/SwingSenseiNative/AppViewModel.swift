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

@MainActor
final class AppViewModel: ObservableObject {
    @Published private(set) var swings: [SwingRecord] = []
    @Published var trimSelection: SwingSelection?
    @Published var viewerSelection: SwingSelection?
    @Published var processingMessage: String?
    @Published var appMessage: AppMessage?

    private let repository: SwingRepository

    init(repository: SwingRepository = SwingRepository()) {
        self.repository = repository
    }

    func loadSwings() {
        do {
            swings = try repository.load()
        } catch {
            appMessage = AppMessage(text: "Could not load swing history: \(error.localizedDescription)")
        }
    }

    func swing(id: String) -> SwingRecord? {
        swings.first { $0.id == id }
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

            try await persistNewSwing(videoURL: pickedVideo.url, source: .uploaded)
        } catch {
            appMessage = AppMessage(text: error.localizedDescription)
        }

        processingMessage = nil
    }

    func finishRecordedVideo(_ videoURL: URL) async {
        do {
            processingMessage = "Saving recording"
            try await persistNewSwing(videoURL: videoURL, source: .recorded)
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

            let analysis = try await client.analyzeSwing(
                videoURL: swing.videoURL,
                swingID: swing.id,
                durationMs: durationMs,
                trimStartMs: trim.startMs,
                trimEndMs: trim.endMs,
                videoEditState: swing.videoEditState
            )

            swing.analysis = analysis
            swing.senseiScore = analysis.senseiScore
            swing.status = .complete
            swing.analysisError = nil
            try save(swing)

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

    private func persistNewSwing(videoURL: URL, source: SwingSource) async throws {
        let durationMs = try await loadDurationMs(videoURL: videoURL)
        let orientation = try await loadOrientation(videoURL: videoURL)
        let swing = try repository.copyVideo(
            from: videoURL,
            source: source,
            durationMs: durationMs,
            orientation: orientation
        )
        try save(swing)
        UINotificationFeedbackGenerator().notificationOccurred(.success)
        trimSelection = SwingSelection(id: swing.id)
    }

    private func save(_ swing: SwingRecord) throws {
        swings = repository.upserting(swing, into: swings)
        try repository.save(swings)
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
