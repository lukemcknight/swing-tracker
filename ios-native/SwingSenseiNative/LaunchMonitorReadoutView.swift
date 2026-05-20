import SwiftUI

struct LaunchMonitorReadoutView<Details: View>: View {
    let swing: SwingRecord
    let analysis: SwingAnalysis
    let onDismiss: () -> Void
    @ViewBuilder let detailsView: () -> Details

    @State private var isShowingDetails = false

    private var club: GolfClub {
        GolfClub.canonical(swing.club) ?? .other
    }

    private var estimate: ShotEstimate? {
        analysis.shotEstimate
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 18) {
                    header
                    heroNumbersGrid
                    missDirectionRow
                    actionButtons
                }
                .padding(20)
            }
            .background(Theme.background.ignoresSafeArea())
            .navigationTitle("Last Swing")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done", action: onDismiss)
                        .foregroundStyle(Theme.primary)
                        .font(.body.weight(.bold))
                }
            }
            .navigationDestination(isPresented: $isShowingDetails) {
                detailsView()
                    .navigationBarBackButtonHidden(false)
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Text(club.rawValue)
                    .font(.headline.weight(.black))
                    .foregroundStyle(.black)
                    .padding(.horizontal, 14)
                    .frame(height: 32)
                    .background(Theme.primary, in: Capsule())

                if let estimate {
                    Text(estimate.confidence.capitalized + " confidence")
                        .font(.caption.weight(.black))
                        .foregroundStyle(confidenceColor(for: estimate.confidence))
                        .padding(.horizontal, 12)
                        .frame(height: 32)
                        .background(confidenceColor(for: estimate.confidence).opacity(0.16), in: Capsule())
                }

                Spacer()
            }

            if let estimate, !estimate.limitations.isEmpty {
                Text(estimate.limitations.first ?? "")
                    .font(.caption)
                    .foregroundStyle(Theme.muted)
                    .lineLimit(2)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var heroNumbersGrid: some View {
        VStack(spacing: 12) {
            HStack(spacing: 12) {
                heroCell(
                    label: "Ball Speed",
                    value: formatMph(estimate?.ballSpeedMphRange?.midpoint),
                    unit: "mph",
                    accent: .primary
                )
                heroCell(
                    label: "Club Speed",
                    value: formatMph(estimate?.clubSpeedMphRange?.midpoint),
                    unit: "mph",
                    accent: .primary
                )
            }
            HStack(spacing: 12) {
                heroCell(
                    label: "Carry",
                    value: formatCarry(),
                    unit: "yd",
                    accent: .accent
                )
                heroCell(
                    label: "Smash",
                    value: formatSmash(),
                    unit: "",
                    accent: .accent
                )
            }
        }
    }

    private var missDirectionRow: some View {
        let zone = (estimate?.landingZone ?? "unknown").lowercased()
        let labelText: String = {
            switch zone {
            case "left": "Pulled left"
            case "right": "Pushed right"
            case "center": "Square contact"
            default: "Direction unclear"
            }
        }()
        let icon: String = {
            switch zone {
            case "left": "arrow.up.left"
            case "right": "arrow.up.right"
            case "center": "arrow.up"
            default: "questionmark"
            }
        }()
        return HStack(spacing: 14) {
            Image(systemName: icon)
                .font(.title3.weight(.black))
                .foregroundStyle(Theme.primary)
                .frame(width: 44, height: 44)
                .background(Theme.primary.opacity(0.12), in: Circle())

            VStack(alignment: .leading, spacing: 3) {
                Text("Miss")
                    .font(.caption.weight(.black))
                    .foregroundStyle(Theme.muted)
                    .textCase(.uppercase)
                    .tracking(1.1)
                Text(labelText)
                    .font(.headline.weight(.black))
                    .foregroundStyle(.white)
            }

            Spacer()
        }
        .padding(16)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Theme.line))
    }

    private var actionButtons: some View {
        VStack(spacing: 12) {
            Button {
                isShowingDetails = true
            } label: {
                Label("See Swing Details", systemImage: "waveform.path.ecg")
                    .font(.headline.weight(.black))
                    .foregroundStyle(.black)
                    .frame(maxWidth: .infinity)
                    .frame(height: 56)
                    .background(Theme.primary, in: Capsule())
            }
            .buttonStyle(.plain)
        }
    }

    private func heroCell(
        label: String,
        value: String,
        unit: String,
        accent: HeroAccent
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(label)
                .font(.caption.weight(.black))
                .foregroundStyle(Theme.muted)
                .textCase(.uppercase)
                .tracking(1.1)

            HStack(alignment: .lastTextBaseline, spacing: 4) {
                Text(value)
                    .font(.system(size: 44, weight: .black, design: .rounded))
                    .foregroundStyle(.white)
                    .minimumScaleFactor(0.6)
                    .lineLimit(1)
                if !unit.isEmpty {
                    Text(unit)
                        .font(.title3.weight(.black))
                        .foregroundStyle(Theme.muted)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 16)
        .padding(.vertical, 18)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(accent == .primary ? Theme.primary.opacity(0.45) : Theme.accent.opacity(0.45), lineWidth: 1.2)
        )
    }

    private enum HeroAccent {
        case primary
        case accent
    }

    private func formatMph(_ value: Double?) -> String {
        guard let value, value > 0 else { return "—" }
        return String(Int(value.rounded()))
    }

    private func formatCarry() -> String {
        guard let ballSpeed = estimate?.ballSpeedMphRange?.midpoint, ballSpeed > 0 else {
            if let clubSpeed = estimate?.clubSpeedMphRange?.midpoint, clubSpeed > 0 {
                let derived = clubSpeed * club.arcadeSmashFactor
                return String(Int(carryFromBallSpeed(derived).rounded()))
            }
            return "—"
        }
        return String(Int(carryFromBallSpeed(ballSpeed).rounded()))
    }

    private func formatSmash() -> String {
        guard
            let ballSpeed = estimate?.ballSpeedMphRange?.midpoint, ballSpeed > 0,
            let clubSpeed = estimate?.clubSpeedMphRange?.midpoint, clubSpeed > 0
        else {
            return "—"
        }
        let ratio = ballSpeed / clubSpeed
        return String(format: "%.2f", ratio)
    }

    private func carryFromBallSpeed(_ ballSpeedMph: Double) -> Double {
        let ratio = max(0.50, min(1.30, ballSpeedMph / club.arcadeReferenceBallSpeedMph))
        return club.arcadeDefaultCarryYards * pow(ratio, 1.16)
    }

    private func confidenceColor(for raw: String) -> Color {
        switch raw.lowercased() {
        case "high":
            Theme.primary
        case "medium":
            Theme.accent
        default:
            Theme.muted
        }
    }
}
