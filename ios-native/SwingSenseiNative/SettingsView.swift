import SwiftUI

struct SettingsView: View {
    @Binding var baseURLString: String

    var body: some View {
        NavigationStack {
            Form {
                Section("Analysis Backend") {
                    TextField("Backend URL", text: $baseURLString)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                        .font(.system(.body, design: .monospaced))

                    Text("Use http://127.0.0.1:8000 for the iOS simulator. Use your Mac's LAN IP for a physical device.")
                        .font(.caption)
                        .foregroundStyle(Theme.muted)
                }

                Section("Native App") {
                    LabeledContent("Stack", value: "SwiftUI + AVFoundation")
                    LabeledContent("Analysis", value: "FastAPI + MediaPipe")
                    LabeledContent("iOS Target", value: "17+")
                }
            }
            .scrollContentBackground(.hidden)
            .background(Theme.background.ignoresSafeArea())
            .navigationTitle("Settings")
        }
    }
}
