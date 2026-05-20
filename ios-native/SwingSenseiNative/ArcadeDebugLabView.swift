#if DEBUG
import SwiftUI
import UIKit

struct ArcadeDebugLabView: View {
    @ObservedObject var viewModel: AppViewModel

    @State private var speedMode: DebugSpeedMode = .ball
    @State private var selectedClub: GolfClub = .sevenIron
    @State private var selectedBallMode: ArcadeBallMode = .realBall
    @State private var selectedHoleIndex = 0
    @State private var selectedLandingZone = "center"
    @State private var selectedConfidence = ArcadeShotConfidence.medium
    @State private var ballSpeedMidpoint = 112.0
    @State private var ballSpeedSpread = 12.0
    @State private var clubSpeedMidpoint = 82.0
    @State private var clubSpeedSpread = 8.0
    @State private var selectedSwingID: String?
    @State private var copiedMessage: String?

    private let shotEngine = ArcadeShotEngine()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                inputSection
                resultSection
                replaySection
                exportSection
            }
            .padding(18)
        }
        .background(Theme.background.ignoresSafeArea())
        .navigationTitle("Arcade Lab")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            if selectedSwingID == nil {
                selectedSwingID = replayableSwings.first?.id
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Debug Only")
                .font(.caption.weight(.black))
                .foregroundStyle(Theme.primary)
                .textCase(.uppercase)
                .tracking(1.2)

            Text("Tune shot results without filming.")
                .font(.system(size: 32, weight: .black, design: .rounded))
                .foregroundStyle(.white)
                .fixedSize(horizontal: false, vertical: true)

            Text("Adjust the estimate inputs, replay saved analyses, and copy round JSON for quick iteration.")
                .font(.body)
                .foregroundStyle(Theme.muted)
                .lineSpacing(3)
        }
    }

    private var inputSection: some View {
        LabPanel(title: "Shot Input") {
            VStack(spacing: 14) {
                Picker("Mode", selection: $speedMode) {
                    ForEach(DebugSpeedMode.allCases) { mode in
                        Text(mode.label).tag(mode)
                    }
                }
                .pickerStyle(.segmented)

                Picker("Hole", selection: $selectedHoleIndex) {
                    ForEach(Array(ArcadeCourseLibrary.starterCourse.holes.enumerated()), id: \.offset) { index, hole in
                        Text("H\(hole.number)").tag(index)
                    }
                }
                .pickerStyle(.segmented)

                Picker("Ball Type", selection: $selectedBallMode) {
                    ForEach(ArcadeBallMode.allCases) { mode in
                        Text(mode.shortLabel).tag(mode)
                    }
                }
                .pickerStyle(.segmented)

                if speedMode != .saved {
                    HStack(spacing: 10) {
                        clubMenu
                        landingZoneMenu
                        confidenceMenu
                    }

                    if speedMode == .ball {
                        speedSlider(
                            title: "Ball speed",
                            value: $ballSpeedMidpoint,
                            spread: $ballSpeedSpread,
                            range: 45...190
                        )
                    } else if speedMode == .club {
                        speedSlider(
                            title: "Club speed",
                            value: $clubSpeedMidpoint,
                            spread: $clubSpeedSpread,
                            range: 35...125
                        )
                    } else {
                        Text("No speed estimate. The engine will use the selected club default and force low confidence.")
                            .font(.callout)
                            .foregroundStyle(Theme.muted)
                            .lineSpacing(3)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                } else {
                    replayPicker
                }
            }
        }
    }

    private var resultSection: some View {
        LabPanel(title: "Resolved Result") {
            VStack(alignment: .leading, spacing: 14) {
                CourseMapView(round: previewRound, hole: result.updatedHole)
                    .frame(height: 300)

                HStack(spacing: 10) {
                    LabMetric(title: "Club", value: result.shot.club)
                    LabMetric(title: "Ball", value: result.shot.ballMode.shortLabel)
                    LabMetric(title: "Carry", value: "\(Int(result.shot.carryYards.rounded())) yd")
                }

                HStack(spacing: 10) {
                    LabMetric(title: "Miss", value: "\(Int(result.shot.lateralMissYards.rounded())) yd")
                    LabMetric(title: "Lie", value: result.shot.lie.label)
                    LabMetric(title: "Putts", value: "\(result.shot.autoPutts)")
                }

                Text(resultSummary)
                    .font(.callout)
                    .foregroundStyle(Theme.muted)
                    .lineSpacing(3)
            }
        }
    }

    private var replaySection: some View {
        LabPanel(title: "Replay Saved Analyses") {
            VStack(alignment: .leading, spacing: 12) {
                if replayableSwings.isEmpty {
                    Text("No analyzed swings yet. Complete at least one analysis, then return here to replay its shot estimate into the course engine.")
                        .font(.callout)
                        .foregroundStyle(Theme.muted)
                        .lineSpacing(3)
                } else {
                    ForEach(Array(replayableSwings.prefix(5))) { swing in
                        Button {
                            speedMode = .saved
                            selectedSwingID = swing.id
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(swing.createdAt.formatted(date: .abbreviated, time: .shortened))
                                        .font(.headline.weight(.black))
                                        .foregroundStyle(.white)

                                    Text(replaySummary(for: swing))
                                        .font(.caption.weight(.bold))
                                        .foregroundStyle(Theme.muted)
                                        .lineLimit(1)
                                }

                                Spacer()

                                if selectedSwingID == swing.id, speedMode == .saved {
                                    Image(systemName: "checkmark.circle.fill")
                                        .font(.title3.weight(.black))
                                        .foregroundStyle(Theme.primary)
                                }
                            }
                            .padding(12)
                            .background(Theme.elevated, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }

    private var exportSection: some View {
        LabPanel(title: "JSON Export") {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 10) {
                    Button {
                        copyToPasteboard(resultJSON, message: "Copied result JSON")
                    } label: {
                        Label("Copy Result", systemImage: "doc.on.doc.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(LabButtonStyle())

                    Button {
                        copyToPasteboard(roundsJSON, message: "Copied rounds JSON")
                    } label: {
                        Label("Copy Rounds", systemImage: "square.and.arrow.up.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(LabButtonStyle())
                    .disabled(viewModel.rounds.isEmpty)
                    .opacity(viewModel.rounds.isEmpty ? 0.45 : 1)
                }

                if let copiedMessage {
                    Text(copiedMessage)
                        .font(.caption.weight(.bold))
                        .foregroundStyle(Theme.primary)
                }

                Text(resultJSON)
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(Theme.muted)
                    .lineLimit(14)
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.black.opacity(0.28), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            }
        }
    }

    private var clubMenu: some View {
        Menu {
            ForEach(GolfClub.groups, id: \.title) { group in
                Section(group.title) {
                    ForEach(group.clubs) { club in
                        Button(club.rawValue) {
                            selectedClub = club
                        }
                    }
                }
            }
        } label: {
            LabMenuLabel(title: "Club", value: selectedClub.rawValue)
        }
    }

    private var landingZoneMenu: some View {
        Menu {
            ForEach(["left", "center", "right", "unknown"], id: \.self) { zone in
                Button(zone.capitalized) {
                    selectedLandingZone = zone
                }
            }
        } label: {
            LabMenuLabel(title: "Zone", value: selectedLandingZone.capitalized)
        }
    }

    private var confidenceMenu: some View {
        Menu {
            ForEach([ArcadeShotConfidence.high, .medium, .low], id: \.rawValue) { confidence in
                Button(confidence.label) {
                    selectedConfidence = confidence
                }
            }
        } label: {
            LabMenuLabel(title: "Conf", value: selectedConfidence.label)
        }
    }

    private var replayPicker: some View {
        VStack(alignment: .leading, spacing: 10) {
            if replayableSwings.isEmpty {
                Text("No saved analyses available.")
                    .font(.callout)
                    .foregroundStyle(Theme.muted)
            } else {
                Menu {
                    ForEach(replayableSwings) { swing in
                        Button(swing.createdAt.formatted(date: .abbreviated, time: .shortened)) {
                            selectedSwingID = swing.id
                        }
                    }
                } label: {
                    LabMenuLabel(title: "Replay", value: replaySwingLabel)
                }

                Text(replayDetail)
                    .font(.callout)
                    .foregroundStyle(Theme.muted)
                    .lineSpacing(3)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private func speedSlider(
        title: String,
        value: Binding<Double>,
        spread: Binding<Double>,
        range: ClosedRange<Double>
    ) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(title)
                    .font(.caption.weight(.black))
                    .foregroundStyle(Theme.primary)
                    .textCase(.uppercase)
                    .tracking(1.1)

                Spacer()

                Text("\(Int((value.wrappedValue - spread.wrappedValue / 2).rounded()))-\(Int((value.wrappedValue + spread.wrappedValue / 2).rounded())) mph")
                    .font(.system(.caption, design: .monospaced).weight(.black))
                    .foregroundStyle(.white)
            }

            Slider(value: value, in: range, step: 1)
                .tint(Theme.primary)

            HStack {
                Text("Range width")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(Theme.muted)

                Slider(value: spread, in: 2...40, step: 2)
                    .tint(Theme.accent)

                Text("\(Int(spread.wrappedValue.rounded()))")
                    .font(.system(.caption, design: .monospaced).weight(.black))
                    .foregroundStyle(Theme.muted)
                    .frame(width: 28, alignment: .trailing)
            }
        }
    }

    private var previewHole: HoleState {
        let holes = ArcadeCourseLibrary.starterCourse.holes
        let index = max(0, min(selectedHoleIndex, holes.count - 1))
        return HoleState(definition: holes[index])
    }

    private var previewRound: RoundRecord {
        var round = RoundRecord.starter(id: "debug-preview")
        if let index = round.holes.firstIndex(where: { $0.id == previewHole.id }) {
            round.holes[index] = result.updatedHole
        }
        return round
    }

    private var result: ArcadeShotResult {
        shotEngine.resolveShot(
            hole: previewHole,
            club: resolvedClub,
            ballMode: selectedBallMode,
            swingID: replaySwing?.id ?? "debug-synthetic",
            shotEstimate: resolvedEstimate,
            createdAt: Date(timeIntervalSince1970: 1_800_000_000),
            shotID: "debug-shot"
        )
    }

    private var resolvedClub: String {
        if speedMode == .saved {
            return replaySwing?.club ?? selectedClub.rawValue
        }
        return selectedClub.rawValue
    }

    private var resolvedEstimate: ShotEstimate? {
        if speedMode == .saved {
            return replaySwing?.analysis?.shotEstimate
        }

        let clubSpeed = speedMode == .club ? SpeedRange(
            min: max(0, clubSpeedMidpoint - clubSpeedSpread / 2),
            max: clubSpeedMidpoint + clubSpeedSpread / 2
        ) : nil
        let ballSpeed = speedMode == .ball ? SpeedRange(
            min: max(0, ballSpeedMidpoint - ballSpeedSpread / 2),
            max: ballSpeedMidpoint + ballSpeedSpread / 2
        ) : nil

        return ShotEstimate(
            clubSpeedMphRange: clubSpeed,
            ballSpeedMphRange: ballSpeed,
            landingZone: selectedLandingZone,
            confidence: selectedConfidence.rawValue,
            evidence: ShotEstimateEvidence(
                sourceFps: 120,
                calibrationSource: "debug_lab",
                detectedBallFlightFrames: speedMode == .ball ? 4 : 0
            ),
            limitations: speedMode == .none ? ["Debug input has no speed estimate."] : []
        )
    }

    private var replayableSwings: [SwingRecord] {
        viewModel.swings.filter { $0.analysis != nil }
    }

    private var replaySwing: SwingRecord? {
        if let selectedSwingID, let swing = replayableSwings.first(where: { $0.id == selectedSwingID }) {
            return swing
        }
        return replayableSwings.first
    }

    private var replaySwingLabel: String {
        guard let replaySwing else { return "None" }
        return replaySwing.createdAt.formatted(date: .abbreviated, time: .omitted)
    }

    private var replayDetail: String {
        guard let replaySwing else { return "No analyzed swing selected." }
        return replaySummary(for: replaySwing)
    }

    private var resultSummary: String {
        let shot = result.shot
        var parts = [
            "\(shot.ballMode.label)",
            "\(shot.confidence.capitalized) confidence",
            "\(Int(shot.distanceDispersionYards.rounded())) yd distance spread",
            "\(Int(shot.lateralDispersionYards.rounded())) yd lateral spread",
        ]
        if shot.autoPutts > 0 {
            parts.append("auto-finished with \(shot.autoPutts) putt\(shot.autoPutts == 1 ? "" : "s")")
        }
        if shot.penaltyStrokes > 0 {
            parts.append("+\(shot.penaltyStrokes) penalty")
        }
        return parts.joined(separator: ", ") + "."
    }

    private var resultJSON: String {
        makeJSON(from: result)
    }

    private var roundsJSON: String {
        makeJSON(from: viewModel.rounds)
    }

    private func replaySummary(for swing: SwingRecord) -> String {
        guard let estimate = swing.analysis?.shotEstimate else {
            return "\(swing.club ?? "Other") / no shot estimate / fallback only"
        }
        let speed: String
        if let ballSpeed = estimate.ballSpeedMphRange {
            speed = "ball \(Int(ballSpeed.min.rounded()))-\(Int(ballSpeed.max.rounded()))"
        } else if let clubSpeed = estimate.clubSpeedMphRange {
            speed = "club \(Int(clubSpeed.min.rounded()))-\(Int(clubSpeed.max.rounded()))"
        } else {
            speed = "no speed"
        }
        return "\(swing.club ?? "Other") / \(speed) / \(estimate.landingZone) / \(estimate.confidence)"
    }

    private func makeJSON<T: Encodable>(from value: T) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(value) else { return "{}" }
        return String(decoding: data, as: UTF8.self)
    }

    private func copyToPasteboard(_ value: String, message: String) {
        UIPasteboard.general.string = value
        copiedMessage = message
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }
}

private enum DebugSpeedMode: String, CaseIterable, Identifiable {
    case ball
    case club
    case none
    case saved

    var id: String { rawValue }

    var label: String {
        switch self {
        case .ball:
            "Ball"
        case .club:
            "Club"
        case .none:
            "None"
        case .saved:
            "Replay"
        }
    }
}

private struct LabPanel<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(title)
                .font(.headline.weight(.black))
                .foregroundStyle(.white)

            content
        }
        .padding(16)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Theme.line))
    }
}

private struct LabMetric: View {
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
                .minimumScaleFactor(0.76)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 11)
        .frame(height: 58)
        .background(Theme.elevated, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

private struct LabMenuLabel: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.caption2.weight(.black))
                .foregroundStyle(Theme.muted)
                .textCase(.uppercase)

            HStack(spacing: 5) {
                Text(value)
                    .font(.headline.weight(.black))
                    .lineLimit(1)
                    .minimumScaleFactor(0.76)

                Image(systemName: "chevron.up.chevron.down")
                    .font(.caption2.weight(.black))
            }
            .foregroundStyle(.white)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 11)
        .frame(height: 58)
        .background(Theme.elevated, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

private struct LabButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.caption.weight(.black))
            .foregroundStyle(.black)
            .frame(height: 46)
            .background(Theme.primary.opacity(configuration.isPressed ? 0.72 : 1), in: Capsule())
    }
}
#endif
