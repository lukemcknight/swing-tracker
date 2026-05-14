import SwiftUI

struct AnalysisViewer: View {
    private static let minimumClubPathAnalysisVersion = 25

    @StateObject private var viewModel: SwingAnalysisViewModel
    @AppStorage("analysisOverlaySelectionV2") private var overlaySelectionStorage = AnalysisOverlayStorage.defaultValue
    @AppStorage("analysisOverlayHelpShownV1") private var hasShownOverlayHelp = false
    @State private var isOverlaySheetPresented = false
    @State private var isOverlayHelpPresented = false
    @State private var selectedClub: GolfClub?
    @State private var aiFeedback: SwingAIFeedback?
    @State private var isRequestingAIFeedback = false
    @State private var aiFeedbackError: String?
    let swingID: String
    let onClubChange: (String) -> Void
    let onRequestAIFeedback: (String) async throws -> SwingAIFeedback
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
        self.swingID = swingID
        self.onClubChange = onClubChange
        self.onRequestAIFeedback = onRequestAIFeedback
        self.onClose = onClose
        _viewModel = StateObject(
            wrappedValue: SwingAnalysisViewModel(
                videoURL: videoURL,
                analysis: analysis,
                videoEditState: videoEditState
            )
        )
        _selectedClub = State(initialValue: GolfClub.canonical(club))
        _aiFeedback = State(initialValue: aiAnalysis)
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                viewerStage

                VStack(spacing: 18) {
                    analysisQualityBanner
                    swingThoughtCard
                    aiAnalysisCard
                    feedbackCards
                    debugPanel
                }
                .padding(.horizontal, 18)
                .padding(.top, 18)
                .padding(.bottom, 28)
            }
        }
        .background(Theme.background.ignoresSafeArea())
        .sheet(isPresented: $isOverlaySheetPresented) {
            OverlaySelectorSheet(
                selectedOverlays: selectedOverlays,
                isAvailable: isOverlayAvailable,
                availabilityMessage: overlayUnavailableReason,
                onToggle: toggleOverlay
            )
            .presentationDetents([.height(430), .medium])
            .presentationDragIndicator(.visible)
            .presentationBackground(Theme.background)
            .presentationCornerRadius(30)
            .onAppear(perform: showOverlayHelpIfNeeded)
            .alert("Overlays", isPresented: $isOverlayHelpPresented) {
                Button("Okay", role: .cancel) {}
            } message: {
                Text("Select or deselect overlays to control what appears on the video.")
            }
        }
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

            scoreBadge
        }
        .padding(.horizontal, 18)
        .frame(height: 86)
        .background(.black)
    }

    private var header: some View {
        HStack {
            Button(action: onClose) {
                Image(systemName: "chevron.left")
                    .font(.headline.weight(.black))
                    .foregroundStyle(.white)
                    .frame(width: 48, height: 48)
                    .background(Theme.surface, in: Circle())
                    .overlay(Circle().stroke(Theme.line))
            }

            Spacer()

            scoreBadge
        }
        .padding(.top, 10)
    }

    private var scoreBadge: some View {
        VStack(spacing: 0) {
            Text(String(format: "%.1f/10", viewModel.analysis.senseiScore))
                .font(.title2.weight(.black))
                .foregroundStyle(.black)

            if isProvisionalScore {
                Text("Provisional")
                    .font(.caption2.weight(.black))
                    .textCase(.uppercase)
                    .foregroundStyle(.black.opacity(0.78))
            }
        }
        .padding(.horizontal, 22)
        .padding(.vertical, isProvisionalScore ? 9 : 12)
        .background(Theme.primary, in: Capsule())
    }

    private var isProvisionalScore: Bool {
        guard let quality = viewModel.analysis.analysisQuality else { return false }
        return quality.status != "ok"
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

            Button {
                isOverlaySheetPresented = true
            } label: {
                Image(systemName: "slider.horizontal.3")
                    .font(.headline.weight(.black))
                    .foregroundStyle(.white)
                    .frame(width: 46, height: 46)
                    .background(.black.opacity(0.68), in: Circle())
                    .overlay(
                        Circle()
                            .stroke(activeOverlays.isEmpty ? Theme.line : Theme.primary.opacity(0.84), lineWidth: 2)
                    )
                    .shadow(color: .black.opacity(0.28), radius: 12, y: 6)
            }
            .buttonStyle(.plain)
            .padding(12)
            .accessibilityLabel("Overlays")
        }
        .aspectRatio(viewModel.videoSize.width / viewModel.videoSize.height, contentMode: .fit)
        .contentShape(Rectangle())
        .onTapGesture {
            viewModel.togglePlay()
        }
    }

    private var analysisControlDeck: some View {
        VStack(spacing: 0) {
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
                    systemName: "point.3.connected.trianglepath.dotted",
                    title: nil,
                    label: "Overlays",
                    action: { isOverlaySheetPresented = true }
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
                    systemName: "ruler",
                    title: nil,
                    label: "Measurements",
                    action: { toggleOverlay(.guides) }
                )

                Spacer()

                toolbarButton(
                    systemName: "trash",
                    title: nil,
                    label: "Clear overlays",
                    color: .red.opacity(0.92),
                    action: clearOptionalOverlays
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

    private var selectedOverlays: Set<AnalysisOverlay> {
        AnalysisOverlayStorage.decode(overlaySelectionStorage)
    }

    private var activeOverlays: Set<AnalysisOverlay> {
        Set(selectedOverlays.filter(isOverlayAvailable))
    }

    private func toggleOverlay(_ overlay: AnalysisOverlay) {
        guard isOverlayAvailable(overlay) else { return }

        var nextSelection = selectedOverlays
        if nextSelection.contains(overlay) {
            nextSelection.remove(overlay)
        } else {
            nextSelection.insert(overlay)
        }
        overlaySelectionStorage = AnalysisOverlayStorage.encode(nextSelection)
    }

    private func clearOptionalOverlays() {
        overlaySelectionStorage = AnalysisOverlayStorage.encode([.joints, .wristPath, .clubPath])
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

    private func isOverlayAvailable(_ overlay: AnalysisOverlay) -> Bool {
        switch overlay {
        case .guides:
            hasAddressGuideData
        case .swingPlane:
            hasSwingPlaneData
        case .clubPath:
            clubHeadPointCount > 0
        case .bodyBox, .headTracking, .spineTracking, .wristPath, .joints:
            true
        }
    }

    private var clubHeadPointCount: Int {
        viewModel.analysis.primaryClubPathSegment()?.samples.count ?? 0
    }

    private var swingPlaneClubHeadPointCount: Int {
        swingPlaneCandidateFrames.count
    }

    private var swingPlaneCandidateFrames: [SwingAnalysisFrame] {
        let framesWithClubHead = viewModel.analysis.primaryClubPathSegment()?.frames ?? []
        guard
            let start = viewModel.analysis.frameIndex(for: .takeaway),
            let end = viewModel.analysis.frameIndex(for: .impact),
            end >= start
        else {
            return framesWithClubHead
        }

        let phaseFrames = viewModel.analysis.primaryClubPathSegment(in: start...end)?.frames ?? []
        return phaseFrames.count >= 3 ? phaseFrames : framesWithClubHead
    }

    private var hasAddressGuideData: Bool {
        guard let address = viewModel.analysis.frame(for: .address) else { return false }
        return hasKeypoint("nose", in: address)
            && hasKeypoint("left_shoulder", in: address)
            && hasKeypoint("right_shoulder", in: address)
            && hasKeypoint("left_hip", in: address)
            && hasKeypoint("right_hip", in: address)
            && (
                hasKeypoint("left_ankle", in: address) && hasKeypoint("right_ankle", in: address)
                || hasKeypoint("left_heel", in: address) && hasKeypoint("right_heel", in: address)
            )
    }

    private var hasSwingPlaneData: Bool {
        swingPlaneClubHeadPointCount >= 3
    }

    private func overlayUnavailableReason(_ overlay: AnalysisOverlay) -> String? {
        guard !isOverlayAvailable(overlay) else { return nil }

        switch overlay {
        case .guides:
            return "Address pose needed"
        case .clubPath:
            return needsReanalysisForClubData ? "Reanalyze swing" : "Club not detected"
        case .swingPlane:
            if needsReanalysisForClubData {
                return "Reanalyze swing"
            }
            if clubHeadPointCount == 0 {
                return "Needs club path"
            }
            return "Need 3 club points"
        case .bodyBox, .headTracking, .spineTracking, .wristPath, .joints:
            return nil
        }
    }

    private var needsReanalysisForClubData: Bool {
        (viewModel.analysis.analysisVersion ?? 0) < Self.minimumClubPathAnalysisVersion && clubHeadPointCount == 0
    }

    private func hasKeypoint(_ name: String, in frame: SwingAnalysisFrame) -> Bool {
        frame.keypoints.contains { $0.name == name && $0.score >= 0.2 }
    }

    private func showOverlayHelpIfNeeded() {
        guard !hasShownOverlayHelp else { return }
        hasShownOverlayHelp = true
        isOverlayHelpPresented = true
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

    private var transportControls: some View {
        VStack(spacing: 12) {
            HStack(spacing: 16) {
                Button {
                    viewModel.stepFrame(-1)
                } label: {
                    Image(systemName: "chevron.left")
                        .font(.title3.weight(.black))
                        .frame(width: 58, height: 58)
                }

                Button {
                    viewModel.togglePlay()
                } label: {
                    Text(viewModel.isPlaying ? "Pause" : "Play")
                        .font(.title2.weight(.black))
                        .frame(maxWidth: .infinity)
                        .frame(height: 58)
                }

                Button {
                    viewModel.stepFrame(1)
                } label: {
                    Image(systemName: "chevron.right")
                        .font(.title3.weight(.black))
                        .frame(width: 58, height: 58)
                }
            }
            .foregroundStyle(.black)
            .buttonStyle(PrimaryControlButtonStyle())

            HStack(spacing: 12) {
                ForEach([Float(0.25), Float(0.5), Float(1.0)], id: \.self) { rate in
                    Button {
                        viewModel.setPlaybackRate(rate)
                    } label: {
                        Text(rateLabel(rate))
                            .font(.headline.weight(.black))
                            .foregroundStyle(viewModel.playbackRate == rate ? .black : .white)
                            .frame(width: 78, height: 48)
                            .background(viewModel.playbackRate == rate ? Theme.primary : Theme.surface, in: Capsule())
                            .overlay(Capsule().stroke(Theme.line))
                    }
                }

                Spacer()

                Text(formatClock(viewModel.playheadMs))
                    .font(.system(.callout, design: .monospaced).weight(.bold))
                    .foregroundStyle(Theme.muted)
            }
        }
    }

    private var scrubber: some View {
        VStack(spacing: 8) {
            Slider(
                value: Binding(
                    get: { viewModel.playheadMs },
                    set: { viewModel.seek(toMs: $0) }
                ),
                in: 0...Double(max(1, viewModel.analysis.durationMs))
            )
            .tint(Theme.primary)

            HStack {
                Text(formatClock(viewModel.playheadMs))
                Spacer()
                Text(formatClock(Double(viewModel.analysis.durationMs)))
            }
            .font(.system(.caption, design: .monospaced).weight(.bold))
            .foregroundStyle(Theme.muted)
        }
        .padding(.horizontal, 4)
    }

    private var feedbackCards: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 14) {
                ForEach(viewModel.analysis.feedback) { item in
                    FeedbackCard(item: item, isActive: item.position == viewModel.selectedPosition)
                }
            }
            .padding(.vertical, 4)
        }
    }

    @ViewBuilder
    private var analysisQualityBanner: some View {
        if let quality = viewModel.analysis.analysisQuality, quality.status != "ok" {
            VStack(alignment: .leading, spacing: 6) {
                Text("Provisional analysis")
                    .font(.caption.weight(.black))
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
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .stroke(Theme.accent.opacity(0.34), lineWidth: 1)
            )
        }
    }

    @ViewBuilder
    private var swingThoughtCard: some View {
        if let swingThought = viewModel.analysis.swingThought {
            VStack(alignment: .leading, spacing: 8) {
                Text("Swing thought")
                    .font(.caption.weight(.black))
                    .foregroundStyle(Theme.primary)
                    .textCase(.uppercase)
                    .tracking(1.5)

                Text(swingThought.thought)
                    .font(.title3.weight(.black))
                    .foregroundStyle(.white)
                    .lineLimit(3)

                Text(swingThought.reason)
                    .font(.callout)
                    .foregroundStyle(Theme.muted)
                    .lineLimit(3)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(18)
            .background(Theme.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .stroke(Theme.primary.opacity(0.45), lineWidth: 1)
            )
        }
    }

    private var aiAnalysisCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 12) {
                Image(systemName: "sparkles")
                    .font(.title3.weight(.black))
                    .foregroundStyle(Theme.primary)
                    .frame(width: 34, height: 34)
                    .background(Theme.primary.opacity(0.12), in: Circle())

                VStack(alignment: .leading, spacing: 2) {
                    Text("AI Analysis")
                        .font(.caption.weight(.black))
                        .foregroundStyle(Theme.primary)
                        .textCase(.uppercase)
                        .tracking(1.5)

                    Text(aiStatusCopy)
                        .font(.caption.weight(.bold))
                        .foregroundStyle(Theme.muted)
                }

                Spacer()

                clubMenu
            }

            if let aiFeedback {
                if isAIAnalysisStale {
                    Text("Club changed since this result was generated.")
                        .font(.callout.weight(.bold))
                        .foregroundStyle(Theme.accent)
                }

                HStack(alignment: .firstTextBaseline, spacing: 12) {
                    Text(String(format: "%.1f", aiFeedback.coachScore))
                        .font(.largeTitle.weight(.black))
                        .foregroundStyle(Theme.primary)
                        .frame(minWidth: 74, alignment: .leading)

                    VStack(alignment: .leading, spacing: 5) {
                        Text(aiFeedback.headline)
                            .font(.title3.weight(.black))
                            .foregroundStyle(.white)
                            .fixedSize(horizontal: false, vertical: true)

                        Text("Confidence: \(aiFeedback.confidence.capitalized)")
                            .font(.caption.weight(.black))
                            .foregroundStyle(Theme.muted)
                    }
                }

                Text(aiFeedback.summary)
                    .font(.callout)
                    .foregroundStyle(.white.opacity(0.9))
                    .fixedSize(horizontal: false, vertical: true)

                AIFeedbackTextSection(title: "Strengths", items: aiFeedback.strengths)
                AIFeedbackFixSection(items: aiFeedback.priorityFixes)
                AIFeedbackDrillSection(items: aiFeedback.practicePlan)
                AIFeedbackTextSection(title: "Limitations", items: aiFeedback.limitations)
            } else {
                Text(selectedClub == nil ? "Select a club before requesting coach-level feedback." : "Ready to compare the video against the app's pose metrics.")
                    .font(.callout)
                    .foregroundStyle(Theme.muted)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if let aiFeedbackError {
                Text(aiFeedbackError)
                    .font(.callout.weight(.bold))
                    .foregroundStyle(.red.opacity(0.92))
                    .fixedSize(horizontal: false, vertical: true)
            }

            Button(action: requestAIFeedback) {
                HStack(spacing: 10) {
                    if isRequestingAIFeedback {
                        ProgressView()
                            .tint(.black)
                    } else {
                        Image(systemName: isAIAnalysisStale ? "arrow.clockwise" : "sparkles")
                            .font(.headline.weight(.black))
                    }

                    Text(aiButtonTitle)
                        .font(.headline.weight(.black))
                }
                .foregroundStyle(.black)
                .frame(maxWidth: .infinity)
                .frame(height: 54)
                .background(canRequestAIFeedback ? Theme.primary : Theme.elevated, in: Capsule())
            }
            .buttonStyle(.plain)
            .disabled(!canRequestAIFeedback)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(18)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(isAIAnalysisStale ? Theme.accent.opacity(0.55) : Theme.primary.opacity(0.45), lineWidth: 1)
        )
    }

    private var clubMenu: some View {
        Menu {
            ForEach(GolfClub.groups, id: \.title) { group in
                Section(group.title) {
                    ForEach(group.clubs) { club in
                        Button {
                            selectClub(club)
                        } label: {
                            if selectedClub == club {
                                Label(club.rawValue, systemImage: "checkmark")
                            } else {
                                Text(club.rawValue)
                            }
                        }
                    }
                }
            }
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "figure.golf")
                    .font(.caption.weight(.black))
                Text(selectedClub?.rawValue ?? "Club")
                    .font(.caption.weight(.black))
                Image(systemName: "chevron.down")
                    .font(.caption2.weight(.black))
            }
            .foregroundStyle(selectedClub == nil ? Theme.accent : .black)
            .padding(.horizontal, 12)
            .frame(height: 34)
            .background(selectedClub == nil ? Theme.accent.opacity(0.14) : Theme.primary, in: Capsule())
            .overlay(Capsule().stroke(selectedClub == nil ? Theme.accent.opacity(0.4) : Theme.primary))
        }
        .disabled(isRequestingAIFeedback)
    }

    private var aiStatusCopy: String {
        guard let aiFeedback else {
            return selectedClub == nil ? "Club required" : "Not run yet"
        }
        return isAIAnalysisStale ? "Needs update" : "\(aiFeedback.model) / \(aiFeedback.club)"
    }

    private var aiButtonTitle: String {
        if isRequestingAIFeedback { return "Analyzing with Gemini" }
        if aiFeedback == nil { return "Get AI Analysis" }
        return "Update AI Analysis"
    }

    private var canRequestAIFeedback: Bool {
        selectedClub != nil && !isRequestingAIFeedback
    }

    private var isAIAnalysisStale: Bool {
        guard let aiFeedback, let selectedClub else { return false }
        if aiFeedback.club != selectedClub.rawValue { return true }
        return aiFeedback.sourceAnalysisVersion != viewModel.analysis.analysisVersion
    }

    private func selectClub(_ club: GolfClub) {
        selectedClub = club
        aiFeedbackError = nil
        onClubChange(club.rawValue)
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }

    private func requestAIFeedback() {
        guard let selectedClub else { return }
        isRequestingAIFeedback = true
        aiFeedbackError = nil
        viewModel.player.pause()

        Task {
            do {
                let feedback = try await onRequestAIFeedback(selectedClub.rawValue)
                aiFeedback = feedback
                if let feedbackClub = GolfClub.canonical(feedback.club) {
                    self.selectedClub = feedbackClub
                }
            } catch {
                aiFeedbackError = error.localizedDescription
                UINotificationFeedbackGenerator().notificationOccurred(.error)
            }
            isRequestingAIFeedback = false
        }
    }

    private var debugPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Debug")
                .font(.caption.weight(.black))
                .foregroundStyle(Theme.primary)
                .textCase(.uppercase)
                .tracking(1.5)

            debugLink("Phase sheet", urlString: viewModel.analysis.debug?.contactSheetUrl)
            debugLink("Impact filmstrip", urlString: viewModel.analysis.debug?.impactFilmstripUrl)

            if let version = viewModel.analysis.analysisVersion {
                Text("Analysis v\(version)")
                    .font(.caption)
                    .foregroundStyle(Theme.muted)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(Theme.line))
    }

    @ViewBuilder
    private func debugLink(_ label: String, urlString: String?) -> some View {
        if let urlString, let url = URL(string: urlString) {
            Link(label, destination: url)
                .font(.callout.weight(.bold))
                .foregroundStyle(.white)
        }
    }

    private func formatClock(_ milliseconds: Double) -> String {
        let totalSeconds = max(0, milliseconds / 1000)
        return String(format: "%.2fs", totalSeconds)
    }

    private func rateLabel(_ rate: Float) -> String {
        switch rate {
        case 0.25:
            "0.25x"
        case 0.5:
            "0.5x"
        default:
            "1x"
        }
    }

    private func cameraAngleLabel(_ angle: String) -> String {
        AnalysisQuality.cameraAngleDisplayName(for: angle)
    }

    private func qualityWarningCopy(_ quality: AnalysisQuality) -> String {
        let warnings = Set(quality.warnings)
        if warnings.contains("low_phase_confidence") {
            return "Some checkpoints were low confidence, so coaching is limited to safer phases."
        }
        if warnings.contains("poor_framing") || warnings.contains("golfer_too_small") {
            return "Frame the full body a little larger next time for stronger checkpoint detection."
        }
        if warnings.contains("unusual_camera_angle") {
            return "Use a clearer down-the-line or face-on view for stronger feedback."
        }
        return "Score and feedback are confidence-gated for this clip."
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

enum AnalysisOverlayStorage {
    static let defaultValue = encode([.joints, .wristPath, .clubPath])

    static func decode(_ rawValue: String) -> Set<AnalysisOverlay> {
        guard !rawValue.isEmpty else { return [] }
        return Set(
            rawValue
                .split(separator: ",")
                .compactMap { AnalysisOverlay(rawValue: String($0)) }
        )
    }

    static func encode(_ overlays: Set<AnalysisOverlay>) -> String {
        AnalysisOverlay.allCases
            .filter { overlays.contains($0) }
            .map(\.rawValue)
            .joined(separator: ",")
    }
}

private struct OverlaySelectorSheet: View {
    let selectedOverlays: Set<AnalysisOverlay>
    let isAvailable: (AnalysisOverlay) -> Bool
    let availabilityMessage: (AnalysisOverlay) -> String?
    let onToggle: (AnalysisOverlay) -> Void

    private let columns = [
        GridItem(.flexible(), spacing: 12),
        GridItem(.flexible(), spacing: 12),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack(spacing: 14) {
                Image(systemName: "slider.horizontal.3")
                    .font(.title2.weight(.black))
                    .foregroundStyle(Theme.primary)
                    .frame(width: 38, height: 38)

                Text("Overlays")
                    .font(.largeTitle.weight(.black))
                    .foregroundStyle(.white)

                Spacer()
            }

            LazyVGrid(columns: columns, spacing: 12) {
                ForEach(AnalysisOverlay.allCases) { overlay in
                    OverlaySelectorTile(
                        title: overlay.title,
                        isSelected: selectedOverlays.contains(overlay) && isAvailable(overlay),
                        isAvailable: isAvailable(overlay),
                        message: availabilityMessage(overlay)
                    ) {
                        onToggle(overlay)
                    }
                }
            }
        }
        .padding(.horizontal, 20)
        .padding(.top, 24)
        .padding(.bottom, 28)
    }
}

private struct OverlaySelectorTile: View {
    let title: String
    let isSelected: Bool
    let isAvailable: Bool
    let message: String?
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.headline.weight(.black))
                        .lineLimit(1)
                        .minimumScaleFactor(0.72)

                    if let message {
                        Text(message)
                            .font(.caption2.weight(.bold))
                            .lineLimit(1)
                            .minimumScaleFactor(0.75)
                    }
                }

                Spacer(minLength: 6)

                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .font(.title2.weight(.bold))
                    .foregroundStyle(checkColor)
            }
            .foregroundStyle(isAvailable ? .white : Theme.muted.opacity(0.64))
            .padding(.horizontal, 14)
            .frame(maxWidth: .infinity)
            .frame(height: 68)
            .background(tileBackground, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .stroke(borderColor, lineWidth: isSelected ? 2 : 1.5)
            )
            .contentShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        }
        .buttonStyle(.plain)
        .disabled(!isAvailable)
        .opacity(isAvailable ? 1 : 0.46)
        .accessibilityLabel(title)
        .accessibilityValue(isSelected ? "Selected" : "Not selected")
    }

    private var tileBackground: Color {
        isSelected ? Theme.primary.opacity(0.16) : Theme.surface
    }

    private var borderColor: Color {
        if !isAvailable { return Theme.line.opacity(0.55) }
        return isSelected ? Theme.primary : Theme.line
    }

    private var checkColor: Color {
        if !isAvailable { return Theme.muted.opacity(0.5) }
        return isSelected ? Theme.primary : Theme.muted.opacity(0.72)
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

private struct PrimaryControlButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .background(Theme.primary, in: Capsule())
            .scaleEffect(configuration.isPressed ? 0.97 : 1)
    }
}

private struct FeedbackCard: View {
    let item: FeedbackItem
    let isActive: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(item.position.label)
                .font(.caption.weight(.black))
                .foregroundStyle(Theme.primary)
                .textCase(.uppercase)
                .tracking(1.5)

            Text(item.label)
                .font(.title2.weight(.black))
                .foregroundStyle(.white)

            Text(item.detail)
                .font(.callout)
                .foregroundStyle(Theme.muted)
                .lineLimit(4)

            Spacer()

            Text(item.status.rawValue.uppercased())
                .font(.caption.weight(.black))
                .foregroundStyle(statusColor)
        }
        .frame(width: 250, height: 190, alignment: .leading)
        .padding(18)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(isActive ? Theme.primary : Theme.line, lineWidth: isActive ? 2 : 1)
        )
    }

    private var statusColor: Color {
        switch item.status {
        case .good:
            Theme.primary
        case .warning:
            Theme.accent
        case .bad:
            .red
        }
    }
}

private struct AIFeedbackTextSection: View {
    let title: String
    let items: [String]

    var body: some View {
        if !items.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                Text(title)
                    .font(.caption.weight(.black))
                    .foregroundStyle(Theme.primary)
                    .textCase(.uppercase)
                    .tracking(1.2)

                ForEach(items, id: \.self) { item in
                    HStack(alignment: .top, spacing: 8) {
                        Circle()
                            .fill(Theme.primary)
                            .frame(width: 5, height: 5)
                            .padding(.top, 7)

                        Text(item)
                            .font(.callout)
                            .foregroundStyle(.white.opacity(0.88))
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        }
    }
}

private struct AIFeedbackFixSection: View {
    let items: [SwingAIPriorityFix]

    var body: some View {
        if !items.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                Text("Priority Fixes")
                    .font(.caption.weight(.black))
                    .foregroundStyle(Theme.primary)
                    .textCase(.uppercase)
                    .tracking(1.2)

                ForEach(items) { item in
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: 8) {
                            Text(item.title)
                                .font(.headline.weight(.black))
                                .foregroundStyle(.white)

                            if let phase = item.phase, !phase.isEmpty {
                                Text(phase)
                                    .font(.caption2.weight(.black))
                                    .foregroundStyle(.black)
                                    .padding(.horizontal, 8)
                                    .frame(height: 22)
                                    .background(Theme.primary, in: Capsule())
                            }
                        }

                        Text(item.observation)
                            .font(.callout)
                            .foregroundStyle(Theme.muted)

                        Text("Cue: \(item.cue)")
                            .font(.callout.weight(.bold))
                            .foregroundStyle(.white.opacity(0.92))
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
                    .background(Theme.elevated, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(Theme.line))
                }
            }
        }
    }
}

private struct AIFeedbackDrillSection: View {
    let items: [SwingAIPracticeDrill]

    var body: some View {
        if !items.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                Text("Practice Plan")
                    .font(.caption.weight(.black))
                    .foregroundStyle(Theme.primary)
                    .textCase(.uppercase)
                    .tracking(1.2)

                ForEach(items) { item in
                    HStack(alignment: .top, spacing: 12) {
                        Text(item.reps)
                            .font(.caption.weight(.black))
                            .foregroundStyle(.black)
                            .padding(.horizontal, 9)
                            .frame(minHeight: 26)
                            .background(Theme.primary, in: Capsule())

                        VStack(alignment: .leading, spacing: 4) {
                            Text(item.title)
                                .font(.callout.weight(.black))
                                .foregroundStyle(.white)

                            Text(item.focus)
                                .font(.callout)
                                .foregroundStyle(Theme.muted)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
            }
        }
    }
}
