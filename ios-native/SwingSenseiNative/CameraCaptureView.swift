import AVFoundation
import SwiftUI
import UIKit

struct CameraCaptureView: View {
    let onCancel: () -> Void
    let onFinished: (URL) async -> Void

    @StateObject private var recorder = CameraRecorder()
    @State private var countdown: Int?
    @State private var isFinishing = false

    var body: some View {
        ZStack {
            CameraPreview(session: recorder.session)
                .ignoresSafeArea()

            Color.black.opacity(recorder.authorizationState == .authorized ? 0 : 0.78)
                .ignoresSafeArea()

            VStack {
                topBar

                Spacer()

                if let countdown {
                    Text("\(countdown)")
                        .font(.system(size: 112, weight: .black, design: .rounded))
                        .foregroundStyle(.white)
                        .shadow(radius: 18)
                        .transition(.scale.combined(with: .opacity))
                }

                Spacer()

                bottomControls
            }
            .padding(20)

            if recorder.authorizationState != .authorized {
                permissionContent
            }
        }
        .task {
            recorder.requestAndConfigure()
        }
        .onDisappear {
            recorder.stopSession()
        }
        .onChange(of: recorder.finishedVideoURL) { _, url in
            guard let url else { return }
            Task {
                isFinishing = true
                await onFinished(url)
                isFinishing = false
                recorder.finishedVideoURL = nil
            }
        }
    }

    private var topBar: some View {
        HStack {
            Button {
                if !recorder.isRecording {
                    onCancel()
                }
            } label: {
                Image(systemName: "xmark")
                    .font(.headline.weight(.black))
                    .foregroundStyle(.white)
                    .frame(width: 48, height: 48)
                    .background(.black.opacity(0.42), in: Circle())
            }
            .disabled(recorder.isRecording || isFinishing)

            Spacer()

            Text(recorder.isRecording ? "Recording" : "Frame the full body and club")
                .font(.callout.weight(.bold))
                .foregroundStyle(.white)
                .padding(.horizontal, 14)
                .frame(height: 42)
                .background(.black.opacity(0.42), in: Capsule())
        }
    }

    private var bottomControls: some View {
        VStack(spacing: 16) {
            if let errorMessage = recorder.errorMessage {
                Text(errorMessage)
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(.red)
                    .padding(12)
                    .background(.black.opacity(0.62), in: RoundedRectangle(cornerRadius: 12))
            }

            Button {
                if recorder.isRecording {
                    UINotificationFeedbackGenerator().notificationOccurred(.success)
                    recorder.stopRecording()
                } else {
                    Task { await startCountdownAndRecord() }
                }
            } label: {
                ZStack {
                    Circle()
                        .stroke(.white.opacity(0.92), lineWidth: 5)
                        .frame(width: 86, height: 86)

                    RoundedRectangle(cornerRadius: recorder.isRecording ? 8 : 34, style: .continuous)
                        .fill(recorder.isRecording ? .red : .white)
                        .frame(width: recorder.isRecording ? 34 : 66, height: recorder.isRecording ? 34 : 66)
                }
            }
            .disabled(countdown != nil || recorder.authorizationState != .authorized || isFinishing)
        }
    }

    private var permissionContent: some View {
        VStack(spacing: 14) {
            Image(systemName: "camera.fill")
                .font(.system(size: 42, weight: .black))
                .foregroundStyle(Theme.primary)

            Text(permissionTitle)
                .font(.title2.weight(.black))
                .foregroundStyle(.white)

            Text(permissionCopy)
                .font(.body)
                .foregroundStyle(Theme.muted)
                .multilineTextAlignment(.center)
                .lineSpacing(3)

            Button("Try Again") {
                recorder.requestAndConfigure()
            }
            .font(.headline.weight(.black))
            .foregroundStyle(.black)
            .padding(.horizontal, 22)
            .frame(height: 52)
            .background(Theme.primary, in: Capsule())
            .padding(.top, 8)
        }
        .padding(28)
    }

    private var permissionTitle: String {
        switch recorder.authorizationState {
        case .checking:
            "Preparing camera"
        case .authorized:
            ""
        case .denied:
            "Camera access needed"
        case .failed:
            "Camera unavailable"
        }
    }

    private var permissionCopy: String {
        switch recorder.authorizationState {
        case .checking:
            "Checking camera and microphone access."
        case .authorized:
            ""
        case .denied:
            "Allow camera and microphone access in Settings to record a swing."
        case .failed:
            recorder.errorMessage ?? "Could not configure the camera on this device."
        }
    }

    private func startCountdownAndRecord() async {
        for value in [3, 2, 1] {
            withAnimation(.spring(response: 0.28, dampingFraction: 0.72)) {
                countdown = value
            }
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
            try? await Task.sleep(nanoseconds: 720_000_000)
        }

        withAnimation(.easeOut(duration: 0.16)) {
            countdown = nil
        }
        UIImpactFeedbackGenerator(style: .heavy).impactOccurred()
        recorder.startRecording(maxDurationSeconds: 30)
    }
}

enum CameraAuthorizationState {
    case checking
    case authorized
    case denied
    case failed
}

final class CameraRecorder: NSObject, ObservableObject {
    let session = AVCaptureSession()

    @Published var authorizationState: CameraAuthorizationState = .checking
    @Published var isRecording = false
    @Published var errorMessage: String?
    @Published var finishedVideoURL: URL?

    private let sessionQueue = DispatchQueue(label: "swing-sensei.camera.session")
    private let movieOutput = AVCaptureMovieFileOutput()
    private var isConfigured = false

    func requestAndConfigure() {
        authorizationState = .checking
        errorMessage = nil

        Task {
            let cameraAllowed = await Self.requestAccess(for: .video)
            let microphoneAllowed = await Self.requestAccess(for: .audio)

            guard cameraAllowed && microphoneAllowed else {
                await MainActor.run {
                    self.authorizationState = .denied
                }
                return
            }

            configureSession()
        }
    }

    func startRecording(maxDurationSeconds: Double) {
        sessionQueue.async {
            guard self.isConfigured, !self.movieOutput.isRecording else { return }

            let destination = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension("mov")
            try? FileManager.default.removeItem(at: destination)

            self.movieOutput.maxRecordedDuration = CMTime(seconds: maxDurationSeconds, preferredTimescale: 600)

            if let connection = self.movieOutput.connection(with: .video) {
                if connection.isVideoStabilizationSupported {
                    connection.preferredVideoStabilizationMode = .cinematic
                }
                if connection.isVideoRotationAngleSupported(90) {
                    connection.videoRotationAngle = 90
                }
            }

            self.movieOutput.startRecording(to: destination, recordingDelegate: self)
        }
    }

    func stopRecording() {
        sessionQueue.async {
            guard self.movieOutput.isRecording else { return }
            self.movieOutput.stopRecording()
        }
    }

    func stopSession() {
        sessionQueue.async {
            if self.movieOutput.isRecording {
                self.movieOutput.stopRecording()
            }
            if self.session.isRunning {
                self.session.stopRunning()
            }
        }
    }

    private func configureSession() {
        sessionQueue.async {
            if self.isConfigured {
                if !self.session.isRunning {
                    self.session.startRunning()
                }
                DispatchQueue.main.async {
                    self.authorizationState = .authorized
                }
                return
            }

            do {
                self.session.beginConfiguration()
                if self.session.canSetSessionPreset(.hd1920x1080) {
                    self.session.sessionPreset = .hd1920x1080
                }

                guard
                    let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                    let videoInput = try? AVCaptureDeviceInput(device: videoDevice),
                    self.session.canAddInput(videoInput)
                else {
                    throw CameraRecorderError.configurationFailed
                }
                self.session.addInput(videoInput)

                if
                    let audioDevice = AVCaptureDevice.default(for: .audio),
                    let audioInput = try? AVCaptureDeviceInput(device: audioDevice),
                    self.session.canAddInput(audioInput)
                {
                    self.session.addInput(audioInput)
                }

                guard self.session.canAddOutput(self.movieOutput) else {
                    throw CameraRecorderError.configurationFailed
                }
                self.session.addOutput(self.movieOutput)
                self.session.commitConfiguration()
                self.isConfigured = true
                self.session.startRunning()

                DispatchQueue.main.async {
                    self.authorizationState = .authorized
                }
            } catch {
                self.session.commitConfiguration()
                DispatchQueue.main.async {
                    self.errorMessage = error.localizedDescription
                    self.authorizationState = .failed
                }
            }
        }
    }

    private static func requestAccess(for mediaType: AVMediaType) async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: mediaType) {
        case .authorized:
            return true
        case .notDetermined:
            return await withCheckedContinuation { continuation in
                AVCaptureDevice.requestAccess(for: mediaType) { allowed in
                    continuation.resume(returning: allowed)
                }
            }
        default:
            return false
        }
    }
}

extension CameraRecorder: AVCaptureFileOutputRecordingDelegate {
    func fileOutput(
        _ output: AVCaptureFileOutput,
        didStartRecordingTo fileURL: URL,
        from connections: [AVCaptureConnection]
    ) {
        DispatchQueue.main.async {
            self.isRecording = true
        }
    }

    func fileOutput(
        _ output: AVCaptureFileOutput,
        didFinishRecordingTo outputFileURL: URL,
        from connections: [AVCaptureConnection],
        error: Error?
    ) {
        let fileExists = FileManager.default.fileExists(atPath: outputFileURL.path)

        DispatchQueue.main.async {
            self.isRecording = false
            if let error, !fileExists {
                self.errorMessage = error.localizedDescription
                UINotificationFeedbackGenerator().notificationOccurred(.error)
                return
            }
            self.finishedVideoURL = outputFileURL
        }
    }
}

private enum CameraRecorderError: Error, LocalizedError {
    case configurationFailed

    var errorDescription: String? {
        "Could not configure camera capture."
    }
}

private struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> CameraPreviewUIView {
        let view = CameraPreviewUIView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: CameraPreviewUIView, context: Context) {
        uiView.previewLayer.session = session
    }
}

private final class CameraPreviewUIView: UIView {
    override static var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }
}
