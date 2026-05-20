import Foundation

enum AnalysisBaseURL {
    static let developmentFallback = "http://127.0.0.1:8000"

    static var resolvedDefault: String {
        let value = Bundle.main.object(forInfoDictionaryKey: "ANALYSIS_BASE_URL") as? String
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? developmentFallback : trimmed
    }
}
