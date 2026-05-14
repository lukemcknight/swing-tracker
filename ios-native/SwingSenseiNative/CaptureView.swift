import PhotosUI
import SwiftUI

struct CaptureView: View {
    @ObservedObject var viewModel: AppViewModel
    @State private var selectedItem: PhotosPickerItem?
    @State private var isCameraPresented = false

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 22) {
                Spacer()

                VStack(alignment: .leading, spacing: 14) {
                    Text("Swing Sensei")
                        .font(.caption.weight(.black))
                        .foregroundStyle(Theme.primary)
                        .textCase(.uppercase)
                        .tracking(1.5)

                    Text("Record. Review. Fix the move.")
                        .font(.system(size: 42, weight: .black, design: .rounded))
                        .foregroundStyle(.white)
                        .fixedSize(horizontal: false, vertical: true)

                    Text("Capture a swing or upload a clip, then step through the key positions with native video controls and feedback.")
                        .font(.body)
                        .foregroundStyle(Theme.muted)
                        .lineSpacing(3)
                }

                Spacer()

                VStack(spacing: 14) {
                    Button {
                        isCameraPresented = true
                    } label: {
                        ActionPanel(
                            title: "Record Swing",
                            subtitle: "Countdown, haptics, and high-quality camera capture.",
                            systemImage: "record.circle.fill",
                            isPrimary: true
                        )
                    }
                    .buttonStyle(.plain)

                    PhotosPicker(selection: $selectedItem, matching: .videos) {
                        ActionPanel(
                            title: "Upload Video",
                            subtitle: "Choose an existing phone video from your library.",
                            systemImage: "photo.on.rectangle.angled",
                            isPrimary: false
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(22)
            .background(Theme.background.ignoresSafeArea())
            .navigationTitle("Capture")
            .navigationBarTitleDisplayMode(.inline)
        }
        .task(id: selectedItem) {
            await viewModel.importPickedVideo(selectedItem)
            selectedItem = nil
        }
        .fullScreenCover(isPresented: $isCameraPresented) {
            CameraCaptureView(
                onCancel: {
                    isCameraPresented = false
                },
                onFinished: { url in
                    isCameraPresented = false
                    await viewModel.finishRecordedVideo(url)
                }
            )
        }
    }
}

private struct ActionPanel: View {
    let title: String
    let subtitle: String
    let systemImage: String
    let isPrimary: Bool

    var body: some View {
        HStack(spacing: 18) {
            Image(systemName: systemImage)
                .font(.system(size: 30, weight: .black))
                .frame(width: 52, height: 52)
                .foregroundStyle(isPrimary ? .black : Theme.primary)
                .background(isPrimary ? .white.opacity(0.22) : Theme.primary.opacity(0.13), in: Circle())

            VStack(alignment: .leading, spacing: 6) {
                Text(title)
                    .font(.title2.weight(.black))
                    .foregroundStyle(isPrimary ? .black : .white)

                Text(subtitle)
                    .font(.callout)
                    .foregroundStyle(isPrimary ? .black.opacity(0.76) : Theme.muted)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer()
        }
        .frame(maxWidth: .infinity, minHeight: 126, alignment: .leading)
        .padding(20)
        .background(isPrimary ? Theme.primary : Theme.surface, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).stroke(isPrimary ? .clear : Theme.line))
    }
}
