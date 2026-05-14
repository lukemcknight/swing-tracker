import PhotosUI
import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = AppViewModel()
    @AppStorage("analysisBaseURL") private var baseURLString = "http://127.0.0.1:8000"

    var body: some View {
        ZStack {
            TabView {
                CaptureView(viewModel: viewModel)
                    .tabItem {
                        Label("Capture", systemImage: "camera.fill")
                    }

                HistoryView(viewModel: viewModel, baseURLString: $baseURLString)
                    .tabItem {
                        Label("History", systemImage: "clock.fill")
                    }

                SettingsView(baseURLString: $baseURLString)
                    .tabItem {
                        Label("Settings", systemImage: "gearshape.fill")
                    }
            }
            .tint(Theme.primary)

            if let processingMessage = viewModel.processingMessage {
                BusyOverlay(message: processingMessage)
            }
        }
        .task {
            viewModel.loadSwings()
        }
        .sheet(item: $viewModel.trimSelection) { selection in
            if let swing = viewModel.swing(id: selection.id) {
                TrimSwingView(
                    swing: swing,
                    onCancel: {
                        viewModel.trimSelection = nil
                    },
                    onAnalyze: { startMs, endMs, durationMs, videoEditState, club in
                        await viewModel.saveTrimAndAnalyze(
                            swingID: swing.id,
                            club: club,
                            startMs: startMs,
                            endMs: endMs,
                            durationMs: durationMs,
                            videoEditState: videoEditState,
                            baseURLString: baseURLString
                        )
                    }
                )
            } else {
                MissingSwingView()
            }
        }
        .fullScreenCover(item: $viewModel.viewerSelection) { selection in
            if
                let swing = viewModel.swing(id: selection.id),
                let analysis = swing.analysis
            {
                AnalysisViewer(
                    swingID: swing.id,
                    videoURL: swing.videoURL,
                    analysis: analysis,
                    club: swing.club,
                    aiAnalysis: swing.aiAnalysis,
                    videoEditState: swing.videoEditState ?? .identity,
                    onClubChange: { club in
                        viewModel.updateClub(swingID: swing.id, club: club)
                    },
                    onRequestAIFeedback: { club in
                        try await viewModel.requestAIFeedback(
                            swingID: swing.id,
                            club: club,
                            baseURLString: baseURLString
                        )
                    }
                ) {
                    viewModel.viewerSelection = nil
                }
            } else {
                MissingSwingView()
            }
        }
        .alert(item: $viewModel.appMessage) { message in
            Alert(title: Text("Swing Sensei"), message: Text(message.text), dismissButton: .default(Text("OK")))
        }
    }
}

private struct BusyOverlay: View {
    let message: String

    var body: some View {
        ZStack {
            Color.black.opacity(0.56)
                .ignoresSafeArea()

            VStack(spacing: 16) {
                ProgressView()
                    .tint(Theme.primary)
                    .scaleEffect(1.35)

                Text(message)
                    .font(.headline.weight(.black))
                    .foregroundStyle(.white)
            }
            .padding(28)
            .background(.black.opacity(0.78), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).stroke(Theme.line))
        }
    }
}

private struct MissingSwingView: View {
    var body: some View {
        VStack(spacing: 12) {
            Text("Swing not found")
                .font(.title2.weight(.black))
                .foregroundStyle(.white)

            Text("The selected swing is no longer available.")
                .font(.callout)
                .foregroundStyle(Theme.muted)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.background.ignoresSafeArea())
    }
}
