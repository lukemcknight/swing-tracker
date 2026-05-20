import SwiftUI

struct PlayView: View {
    @ObservedObject var viewModel: AppViewModel
    @Binding var baseURLString: String
    @State private var selectedClub: GolfClub = .sevenIron
    @State private var selectedBallMode: ArcadeBallMode = .realBall
    @State private var isCameraPresented = false
    @State private var isAbandonConfirmationPresented = false
    @State private var isRestartConfirmationPresented = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    if let round = viewModel.activeRound {
                        if round.status == .inProgress, let hole = round.currentHole {
                            activeRoundContent(round: round, hole: hole)
                        } else {
                            completedRoundContent(round: round)
                        }
                    } else {
                        startContent
                    }

                    recentRounds
                }
                .padding(18)
            }
            .background(Theme.background.ignoresSafeArea())
            .navigationTitle("Play")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                if let round = viewModel.activeRound, round.status == .inProgress {
                    ToolbarItem(placement: .topBarLeading) {
                        Menu {
                            Button {
                                isRestartConfirmationPresented = true
                            } label: {
                                Label("Restart Round", systemImage: "arrow.counterclockwise")
                            }

                            Button(role: .destructive) {
                                isAbandonConfirmationPresented = true
                            } label: {
                                Label("Abandon Round", systemImage: "xmark.circle")
                            }
                        } label: {
                            Image(systemName: "ellipsis.circle")
                                .font(.headline.weight(.black))
                        }
                        .accessibilityLabel("Round Options")
                    }
                }
                #if DEBUG
                ToolbarItem(placement: .topBarTrailing) {
                    NavigationLink {
                        ArcadeDebugLabView(viewModel: viewModel)
                    } label: {
                        Image(systemName: "slider.horizontal.3")
                            .font(.headline.weight(.black))
                    }
                    .accessibilityLabel("Arcade Debug Lab")
                }
                #endif
            }
            .confirmationDialog(
                "Restart this round?",
                isPresented: $isRestartConfirmationPresented,
                titleVisibility: .visible
            ) {
                Button("Restart", role: .destructive) {
                    viewModel.restartActiveRound()
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("All shots on every hole will be cleared.")
            }
            .confirmationDialog(
                "Abandon this round?",
                isPresented: $isAbandonConfirmationPresented,
                titleVisibility: .visible
            ) {
                Button("Abandon", role: .destructive) {
                    viewModel.abandonActiveRound()
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("You'll return to the start screen. The round will be saved as ended.")
            }
        }
        .fullScreenCover(isPresented: $isCameraPresented) {
            CameraCaptureView(
                onCancel: {
                    isCameraPresented = false
                },
                onFinished: { url in
                    isCameraPresented = false
                    await viewModel.finishRecordedRoundShotVideo(
                        url,
                        club: selectedClub.rawValue,
                        ballMode: selectedBallMode
                    )
                }
            )
        }
    }

    private func activeRoundContent(round: RoundRecord, hole: HoleState) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            RoundHeader(round: round, hole: hole)

            CourseMapView(round: round, hole: hole)
                .frame(height: 380)

            if let pendingSwingID = viewModel.pendingRoundShotSwingID,
               let pendingSwing = viewModel.swing(id: pendingSwingID),
               pendingSwing.status == .failed {
                PendingShotRetryPanel(
                    error: pendingSwing.analysisError,
                    onRetry: {
                        Task { await viewModel.retryPendingRoundShot(baseURLString: baseURLString) }
                    },
                    onDiscard: {
                        viewModel.discardPendingRoundShot()
                    }
                )
            }

            ShotControlPanel(
                hole: hole,
                selectedClub: $selectedClub,
                selectedBallMode: $selectedBallMode,
                onRecord: {
                    isCameraPresented = true
                }
            )

            if let shot = round.latestShot {
                ShotResultPanel(
                    shot: shot,
                    onRedo: {
                        viewModel.undoLastRoundShot()
                    }
                )
            }
        }
    }

    private func completedRoundContent(round: RoundRecord) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 10) {
                Text(round.courseName)
                    .font(.caption.weight(.black))
                    .foregroundStyle(Theme.primary)
                    .textCase(.uppercase)
                    .tracking(1.2)

                Text("Round Complete")
                    .font(.system(size: 36, weight: .black, design: .rounded))
                    .foregroundStyle(.white)

                Text("\(round.totalStrokes) strokes on \(round.holes.count) holes")
                    .font(.title3.weight(.bold))
                    .foregroundStyle(Theme.muted)
            }

            ScorecardView(round: round)

            Button {
                viewModel.startStarterRound()
            } label: {
                Label("Start New Round", systemImage: "flag.fill")
                    .font(.headline.weight(.black))
                    .foregroundStyle(.black)
                    .frame(maxWidth: .infinity)
                    .frame(height: 56)
                    .background(Theme.primary, in: Capsule())
            }
            .buttonStyle(.plain)
        }
    }

    private var startContent: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 12) {
                Text("Arcade Course")
                    .font(.caption.weight(.black))
                    .foregroundStyle(Theme.primary)
                    .textCase(.uppercase)
                    .tracking(1.2)

                Text("Play golf with your real swing.")
                    .font(.system(size: 40, weight: .black, design: .rounded))
                    .foregroundStyle(.white)
                    .fixedSize(horizontal: false, vertical: true)

                Text("Record each shot, trim the swing, and the app turns the estimate into a simple course result.")
                    .font(.body)
                    .foregroundStyle(Theme.muted)
                    .lineSpacing(3)
            }

            StarterCoursePreview()

            Button {
                viewModel.startStarterRound()
            } label: {
                Label("Start 3-Hole Round", systemImage: "play.fill")
                    .font(.headline.weight(.black))
                    .foregroundStyle(.black)
                    .frame(maxWidth: .infinity)
                    .frame(height: 58)
                    .background(Theme.primary, in: Capsule())
            }
            .buttonStyle(.plain)
        }
        .padding(.top, 36)
    }

    @ViewBuilder
    private var recentRounds: some View {
        if !viewModel.rounds.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                Text("Rounds")
                    .font(.caption.weight(.black))
                    .foregroundStyle(Theme.primary)
                    .textCase(.uppercase)
                    .tracking(1.2)

                ForEach(viewModel.rounds.prefix(4)) { round in
                    Button {
                        viewModel.continueRound(round)
                    } label: {
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(round.startedAt.formatted(date: .abbreviated, time: .shortened))
                                    .font(.headline.weight(.black))
                                    .foregroundStyle(.white)

                                Text("\(round.courseName) / \(round.status.label)")
                                    .font(.caption.weight(.bold))
                                    .foregroundStyle(Theme.muted)
                            }

                            Spacer()

                            Text("\(round.totalStrokes)")
                                .font(.title3.weight(.black))
                                .foregroundStyle(Theme.primary)
                        }
                        .padding(14)
                        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Theme.line))
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.top, 8)
        }
    }
}

private struct RoundHeader: View {
    let round: RoundRecord
    let hole: HoleState

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(round.courseName)
                        .font(.caption.weight(.black))
                        .foregroundStyle(Theme.primary)
                        .textCase(.uppercase)
                        .tracking(1.2)

                    Text("Hole \(hole.holeNumber)")
                        .font(.system(size: 34, weight: .black, design: .rounded))
                        .foregroundStyle(.white)
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 4) {
                    Text("Score")
                        .font(.caption.weight(.black))
                        .foregroundStyle(Theme.muted)
                        .textCase(.uppercase)

                    Text("\(round.totalStrokes)")
                        .font(.title.weight(.black))
                        .foregroundStyle(Theme.primary)
                }
            }

            HStack(spacing: 10) {
                StatPill(title: "Par", value: "\(hole.par)")
                StatPill(title: "To Pin", value: "\(Int(hole.distanceRemainingYards.rounded())) yd")
                StatPill(title: "Lie", value: hole.lie.label)
            }
        }
    }
}

struct CourseMapView: View {
    let round: RoundRecord
    let hole: HoleState

    var body: some View {
        GeometryReader { geometry in
            let size = geometry.size
            let scale = CourseMapScale(size: size, hole: hole)

            ZStack {
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(Color(red: 0.10, green: 0.19, blue: 0.12))

                fairway(scale: scale)
                green(scale: scale)
                tee(scale: scale)
                shotPaths(scale: scale)
                ball(scale: scale)
                    .animation(.easeOut(duration: 0.7), value: hole.shots.count)
                pin(scale: scale)

                VStack {
                    HStack {
                        Text("\(Int(hole.lengthYards.rounded())) yd")
                            .font(.caption.weight(.black))
                            .foregroundStyle(.white.opacity(0.86))
                            .padding(.horizontal, 10)
                            .frame(height: 30)
                            .background(.black.opacity(0.32), in: Capsule())

                        Spacer()

                        Text("\(round.completedHoles)/\(round.holes.count)")
                            .font(.caption.weight(.black))
                            .foregroundStyle(.white.opacity(0.86))
                            .padding(.horizontal, 10)
                            .frame(height: 30)
                            .background(.black.opacity(0.32), in: Capsule())
                    }
                    Spacer()
                }
                .padding(14)
            }
            .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).stroke(Theme.line))
        }
    }

    private func fairway(scale: CourseMapScale) -> some View {
        Capsule()
            .fill(Color(red: 0.23, green: 0.50, blue: 0.22))
            .frame(
                width: max(28, scale.xLength(hole.fairwayWidthYards)),
                height: max(80, scale.yLength(hole.lengthYards))
            )
            .position(
                x: scale.point(ArcadePoint(x: 0, y: hole.lengthYards / 2)).x,
                y: scale.point(ArcadePoint(x: 0, y: hole.lengthYards / 2)).y
            )
    }

    private func green(scale: CourseMapScale) -> some View {
        Circle()
            .fill(Color(red: 0.32, green: 0.72, blue: 0.29))
            .frame(
                width: max(40, scale.xLength(hole.greenRadiusYards * 2)),
                height: max(40, scale.xLength(hole.greenRadiusYards * 2))
            )
            .position(scale.point(hole.greenCenter))
    }

    private func tee(scale: CourseMapScale) -> some View {
        RoundedRectangle(cornerRadius: 6, style: .continuous)
            .fill(Color(red: 0.18, green: 0.32, blue: 0.18))
            .frame(width: 54, height: 18)
            .position(scale.point(ArcadePoint(x: 0, y: 0)))
    }

    private func shotPaths(scale: CourseMapScale) -> some View {
        ZStack {
            ForEach(hole.shots) { shot in
                Path { path in
                    path.move(to: scale.point(shot.startPosition))
                    path.addLine(to: scale.point(shot.endPosition))
                }
                .stroke(.white.opacity(0.72), style: StrokeStyle(lineWidth: 3, lineCap: .round, dash: [7, 7]))
            }
        }
    }

    private func ball(scale: CourseMapScale) -> some View {
        Circle()
            .fill(.white)
            .frame(width: 15, height: 15)
            .shadow(color: .black.opacity(0.36), radius: 5, y: 2)
            .position(scale.point(hole.ballPosition))
    }

    private func pin(scale: CourseMapScale) -> some View {
        ZStack(alignment: .top) {
            Rectangle()
                .fill(.white)
                .frame(width: 3, height: 30)
                .offset(y: 10)

            Image(systemName: "flag.fill")
                .font(.title3.weight(.black))
                .foregroundStyle(Theme.accent)
                .offset(x: 8)
        }
        .position(scale.point(hole.greenCenter))
    }
}

private struct PendingShotRetryPanel: View {
    let error: String?
    let onRetry: () -> Void
    let onDiscard: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.title3.weight(.black))
                    .foregroundStyle(Theme.accent)

                Text("Last shot didn't analyze")
                    .font(.headline.weight(.black))
                    .foregroundStyle(.white)

                Spacer()
            }

            if let error, !error.isEmpty {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(Theme.muted)
                    .lineLimit(3)
            }

            HStack(spacing: 10) {
                Button(action: onRetry) {
                    Label("Retry", systemImage: "arrow.clockwise")
                        .font(.subheadline.weight(.black))
                        .foregroundStyle(.black)
                        .frame(maxWidth: .infinity)
                        .frame(height: 44)
                        .background(Theme.primary, in: Capsule())
                }
                .buttonStyle(.plain)

                Button(action: onDiscard) {
                    Text("Discard")
                        .font(.subheadline.weight(.black))
                        .foregroundStyle(Theme.muted)
                        .frame(maxWidth: .infinity)
                        .frame(height: 44)
                        .background(Theme.elevated, in: Capsule())
                        .overlay(Capsule().stroke(Theme.line))
                }
                .buttonStyle(.plain)
            }
        }
        .padding(16)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Theme.accent.opacity(0.5)))
    }
}

private struct ShotControlPanel: View {
    let hole: HoleState
    @Binding var selectedClub: GolfClub
    @Binding var selectedBallMode: ArcadeBallMode
    let onRecord: () -> Void

    var body: some View {
        VStack(spacing: 14) {
            HStack(spacing: 12) {
                Image(systemName: "figure.golf")
                    .font(.title3.weight(.black))
                    .foregroundStyle(Theme.primary)
                    .frame(width: 42, height: 42)
                    .background(Theme.primary.opacity(0.12), in: Circle())

                VStack(alignment: .leading, spacing: 3) {
                    Text("Next Shot")
                        .font(.caption.weight(.black))
                        .foregroundStyle(Theme.primary)
                        .textCase(.uppercase)
                        .tracking(1.1)

                    Text("\(Int(hole.distanceRemainingYards.rounded())) yards from \(hole.lie.label.lowercased())")
                        .font(.callout)
                        .foregroundStyle(Theme.muted)
                }

                Spacer()

                clubMenu
            }

            Picker("Ball Type", selection: $selectedBallMode) {
                ForEach(ArcadeBallMode.allCases) { mode in
                    Text(mode.shortLabel).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            Text(ballModeHint(for: selectedBallMode))
                .font(.caption)
                .foregroundStyle(Theme.muted)
                .frame(maxWidth: .infinity, alignment: .leading)
                .lineSpacing(2)

            Button(action: onRecord) {
                Label("Record Shot", systemImage: "record.circle.fill")
                    .font(.headline.weight(.black))
                    .foregroundStyle(.black)
                    .frame(maxWidth: .infinity)
                    .frame(height: 56)
                    .background(Theme.primary, in: Capsule())
            }
            .buttonStyle(.plain)
        }
        .padding(16)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Theme.line))
    }

    private func ballModeHint(for mode: ArcadeBallMode) -> String {
        switch mode {
        case .realBall:
            "Real ball — most accurate carry and direction."
        case .foamBall:
            "Foam ball — speed is estimated from club, distance scaled down."
        case .noBall:
            "No ball — speed only, lots of dispersion. Use for practice swings."
        }
    }

    private var clubMenu: some View {
        Menu {
            ForEach(GolfClub.groups, id: \.title) { group in
                Section(group.title) {
                    ForEach(group.clubs) { club in
                        Button {
                            selectedClub = club
                        } label: {
                            if club == selectedClub {
                                Label(club.rawValue, systemImage: "checkmark")
                            } else {
                                Text(club.rawValue)
                            }
                        }
                    }
                }
            }
        } label: {
            HStack(spacing: 7) {
                Text(selectedClub.rawValue)
                    .font(.headline.weight(.black))
                Image(systemName: "chevron.up.chevron.down")
                    .font(.caption.weight(.black))
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 14)
            .frame(height: 42)
            .background(Theme.elevated, in: Capsule())
            .overlay(Capsule().stroke(Theme.line))
        }
    }
}

private struct ShotResultPanel: View {
    let shot: RoundShot
    let onRedo: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Last Shot")
                    .font(.headline.weight(.black))
                    .foregroundStyle(.white)

                Spacer()

                Text(shot.confidence.capitalized)
                    .font(.caption.weight(.black))
                    .foregroundStyle(confidenceColor)
                    .padding(.horizontal, 12)
                    .frame(height: 30)
                    .background(confidenceColor.opacity(0.16), in: Capsule())
            }

            HStack(spacing: 10) {
                StatPill(title: "Club", value: shot.club)
                StatPill(title: "Ball", value: shot.ballMode.shortLabel)
                StatPill(title: "Carry", value: "\(Int(shot.carryYards.rounded())) yd")
            }

            Text(resultLine)
                .font(.callout)
                .foregroundStyle(Theme.muted)
                .lineSpacing(3)

            Button(action: onRedo) {
                Label("Redo Last Shot", systemImage: "arrow.uturn.backward")
                    .font(.subheadline.weight(.black))
                    .foregroundStyle(Theme.primary)
                    .frame(maxWidth: .infinity)
                    .frame(height: 44)
                    .background(Theme.primary.opacity(0.12), in: Capsule())
            }
            .buttonStyle(.plain)
        }
        .padding(16)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Theme.line))
    }

    private var resultLine: String {
        var parts = [
            "Landed \(shot.lie.label.lowercased())",
            "Arcade range: \(Int(shot.distanceDispersionYards.rounded())) yards long/short",
            "\(Int(shot.lateralDispersionYards.rounded())) yards left/right",
        ]
        if shot.autoPutts > 0 {
            parts.append("auto-finished with \(shot.autoPutts) putt\(shot.autoPutts == 1 ? "" : "s")")
        }
        if shot.penaltyStrokes > 0 {
            parts.append("+\(shot.penaltyStrokes) penalty")
        }
        return parts.joined(separator: ", ") + "."
    }

    private var confidenceColor: Color {
        switch ArcadeShotConfidence(shot.confidence) {
        case .high:
            Theme.primary
        case .medium:
            Theme.accent
        case .low:
            Theme.muted
        }
    }
}

private struct ScorecardView: View {
    let round: RoundRecord

    var body: some View {
        VStack(spacing: 0) {
            ForEach(round.holes) { hole in
                HStack {
                    Text("Hole \(hole.holeNumber)")
                        .font(.headline.weight(.black))
                        .foregroundStyle(.white)

                    Spacer()

                    Text("Par \(hole.par)")
                        .font(.callout.weight(.bold))
                        .foregroundStyle(Theme.muted)

                    Text("\(hole.strokes)")
                        .font(.title3.weight(.black))
                        .foregroundStyle(scoreColor(for: hole))
                        .frame(width: 44, alignment: .trailing)
                }
                .padding(.vertical, 13)

                if hole.id != round.holes.last?.id {
                    Divider().overlay(Theme.line)
                }
            }
        }
        .padding(.horizontal, 16)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Theme.line))
    }

    private func scoreColor(for hole: HoleState) -> Color {
        if hole.strokes <= hole.par {
            return Theme.primary
        }
        if hole.strokes == hole.par + 1 {
            return Theme.accent
        }
        return .red
    }
}

private struct StarterCoursePreview: View {
    private let course = ArcadeCourseLibrary.starterCourse

    var body: some View {
        VStack(spacing: 0) {
            ForEach(course.holes) { hole in
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Hole \(hole.number)")
                            .font(.headline.weight(.black))
                            .foregroundStyle(.white)

                        Text("Par \(hole.par)")
                            .font(.caption.weight(.bold))
                            .foregroundStyle(Theme.muted)
                    }

                    Spacer()

                    Text("\(Int(hole.lengthYards.rounded())) yd")
                        .font(.headline.weight(.black))
                        .foregroundStyle(Theme.primary)
                }
                .padding(.vertical, 13)

                if hole.id != course.holes.last?.id {
                    Divider().overlay(Theme.line)
                }
            }
        }
        .padding(.horizontal, 16)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Theme.line))
    }
}

private struct StatPill: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.caption2.weight(.black))
                .foregroundStyle(Theme.muted)
                .textCase(.uppercase)

            Text(value)
                .font(.headline.weight(.black))
                .foregroundStyle(.white)
                .lineLimit(1)
                .minimumScaleFactor(0.78)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12)
        .frame(height: 64)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Theme.line))
    }
}

private struct CourseMapScale {
    let size: CGSize
    let hole: HoleState

    private var horizontalPadding: CGFloat { 28 }
    private var verticalPadding: CGFloat { 34 }

    func point(_ point: ArcadePoint) -> CGPoint {
        let xScale = (size.width - horizontalPadding * 2) / CGFloat(max(1, hole.outOfBoundsWidthYards))
        let yScale = (size.height - verticalPadding * 2) / CGFloat(max(1, hole.lengthYards + 28))
        let centerX = size.width / 2
        let bottomY = size.height - verticalPadding
        return CGPoint(
            x: centerX + CGFloat(point.x) * xScale,
            y: bottomY - CGFloat(point.y) * yScale
        )
    }

    func xLength(_ yards: Double) -> CGFloat {
        let xScale = (size.width - horizontalPadding * 2) / CGFloat(max(1, hole.outOfBoundsWidthYards))
        return CGFloat(yards) * xScale
    }

    func yLength(_ yards: Double) -> CGFloat {
        let yScale = (size.height - verticalPadding * 2) / CGFloat(max(1, hole.lengthYards + 28))
        return CGFloat(yards) * yScale
    }
}

private extension RoundRecord {
    var latestShot: RoundShot? {
        holes.flatMap(\.shots).sorted { $0.createdAt < $1.createdAt }.last
    }
}
