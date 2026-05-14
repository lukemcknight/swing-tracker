import AVFoundation
import SwiftUI
import UIKit

struct HistoryView: View {
    @ObservedObject var viewModel: AppViewModel
    @Binding var baseURLString: String

    private let columns = [
        GridItem(.flexible(), spacing: 12),
        GridItem(.flexible(), spacing: 12),
    ]

    var body: some View {
        NavigationStack {
            Group {
                if viewModel.swings.isEmpty {
                    emptyState
                } else {
                    ScrollView {
                        LazyVGrid(columns: columns, spacing: 12) {
                            ForEach(viewModel.swings) { swing in
                                Button {
                                    viewModel.open(swing)
                                } label: {
                                    SwingHistoryCard(swing: swing)
                                }
                                .buttonStyle(.plain)
                                .contextMenu {
                                    if swing.status == .complete {
                                        Button("Reanalyze") {
                                            Task {
                                                await viewModel.retryAnalysis(swingID: swing.id, baseURLString: baseURLString)
                                            }
                                        }
                                    } else if swing.status == .failed || swing.status == .pending {
                                        Button("Retry Analysis") {
                                            Task {
                                                await viewModel.retryAnalysis(swingID: swing.id, baseURLString: baseURLString)
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        .padding(16)
                    }
                }
            }
            .background(Theme.background.ignoresSafeArea())
            .navigationTitle("History")
        }
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "figure.golf")
                .font(.system(size: 56, weight: .black))
                .foregroundStyle(Theme.primary)

            Text("No swings yet")
                .font(.title.weight(.black))
                .foregroundStyle(.white)

            Text("Record or upload a swing to start building your library.")
                .font(.body)
                .foregroundStyle(Theme.muted)
                .multilineTextAlignment(.center)
                .lineSpacing(3)
                .padding(.horizontal, 32)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct SwingHistoryCard: View {
    let swing: SwingRecord

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ZStack(alignment: .topTrailing) {
                SwingThumbnail(url: swing.videoURL)
                    .aspectRatio(0.72, contentMode: .fill)
                    .frame(maxWidth: .infinity)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))

                scoreBadge
                    .padding(8)
            }

            Text(swing.createdAt.formatted(date: .abbreviated, time: .shortened))
                .font(.headline.weight(.black))
                .foregroundStyle(.white)
                .lineLimit(1)

            HStack(spacing: 6) {
                Text(swing.source.label)
                Text("•")
                Text(swing.status.label)
            }
            .font(.caption.weight(.bold))
            .foregroundStyle(statusColor)
            .lineLimit(1)
        }
        .padding(10)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Theme.line))
    }

    @ViewBuilder
    private var scoreBadge: some View {
        if let score = swing.senseiScore {
            Text(String(format: "%.1f", score))
                .font(.caption.weight(.black))
                .foregroundStyle(.black)
                .frame(minWidth: 42)
                .frame(height: 30)
                .background(Theme.primary, in: Capsule())
        } else {
            Text("--")
                .font(.caption.weight(.black))
                .foregroundStyle(Theme.muted)
                .frame(minWidth: 42)
                .frame(height: 30)
                .background(.black.opacity(0.62), in: Capsule())
        }
    }

    private var statusColor: Color {
        switch swing.status {
        case .complete:
            Theme.primary
        case .failed:
            .red
        case .uploading, .analyzing:
            Theme.accent
        default:
            Theme.muted
        }
    }
}

struct SwingThumbnail: View {
    let url: URL
    @State private var image: UIImage?

    var body: some View {
        ZStack {
            if let image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
            } else {
                Rectangle()
                    .fill(Theme.elevated)
                    .overlay {
                        Image(systemName: "play.fill")
                            .font(.title.weight(.black))
                            .foregroundStyle(Theme.primary)
                    }
            }
        }
        .task(id: url) {
            image = await makeThumbnail(url: url)
        }
    }

    private func makeThumbnail(url: URL) async -> UIImage? {
        await Task.detached(priority: .utility) {
            let asset = AVURLAsset(url: url)
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            generator.maximumSize = CGSize(width: 420, height: 420)

            do {
                let image = try generator.copyCGImage(at: CMTime(seconds: 0.15, preferredTimescale: 600), actualTime: nil)
                return UIImage(cgImage: image)
            } catch {
                return nil
            }
        }.value
    }
}
