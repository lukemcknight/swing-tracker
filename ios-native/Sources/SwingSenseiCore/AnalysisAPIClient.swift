import Foundation

public enum AnalysisAPIError: Error, LocalizedError {
    case invalidBaseURL(String)
    case invalidResponse
    case requestFailed(statusCode: Int, message: String)

    public var errorDescription: String? {
        switch self {
        case let .invalidBaseURL(value):
            "Invalid backend URL: \(value)"
        case .invalidResponse:
            "The backend returned an invalid response."
        case let .requestFailed(statusCode, message):
            message.isEmpty ? "Analysis failed with status \(statusCode)." : message
        }
    }
}

public final class AnalysisAPIClient {
    public let baseURL: URL
    private let session: URLSession
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder

    public init(baseURL: URL = URL(string: "http://127.0.0.1:8000")!, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.session = session
        self.decoder = JSONDecoder()
        self.encoder = JSONEncoder()
    }

    public convenience init(baseURLString: String) throws {
        guard let url = URL(string: baseURLString), url.scheme != nil else {
            throw AnalysisAPIError.invalidBaseURL(baseURLString)
        }
        self.init(baseURL: url)
    }

    public func analyzeSwing(
        videoURL: URL,
        swingID: String,
        club: String? = nil,
        durationMs: Int?,
        trimStartMs: Int? = nil,
        trimEndMs: Int? = nil,
        videoEditState: VideoEditState? = nil
    ) async throws -> SwingAnalysis {
        let videoData = try Data(contentsOf: videoURL)
        let request = try makeAnalyzeRequest(
            swingID: swingID,
            club: club,
            durationMs: durationMs,
            trimStartMs: trimStartMs,
            trimEndMs: trimEndMs,
            videoEditState: videoEditState,
            videoData: videoData,
            filename: Self.uploadFilename(swingID: swingID, videoURL: videoURL),
            mimeType: Self.mimeType(for: videoURL)
        )

        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw AnalysisAPIError.invalidResponse
        }

        guard 200..<300 ~= httpResponse.statusCode else {
            throw AnalysisAPIError.requestFailed(
                statusCode: httpResponse.statusCode,
                message: Self.errorMessage(from: data)
            )
        }

        return try decoder.decode(SwingAnalysis.self, from: data)
            .normalizingDebugURLs(baseURL: baseURL)
    }

    public func requestAIFeedback(
        videoURL: URL,
        swingID: String,
        club: String,
        analysis: SwingAnalysis,
        durationMs: Int?,
        trimStartMs: Int? = nil,
        trimEndMs: Int? = nil,
        videoEditState: VideoEditState? = nil
    ) async throws -> SwingAIFeedback {
        let videoData = try Data(contentsOf: videoURL)
        let analysisJSON = String(decoding: try encoder.encode(analysis), as: UTF8.self)
        let request = try makeAIFeedbackRequest(
            swingID: swingID,
            club: club,
            analysisJSON: analysisJSON,
            durationMs: durationMs,
            trimStartMs: trimStartMs,
            trimEndMs: trimEndMs,
            videoEditState: videoEditState,
            videoData: videoData,
            filename: Self.uploadFilename(swingID: swingID, videoURL: videoURL),
            mimeType: Self.mimeType(for: videoURL)
        )

        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw AnalysisAPIError.invalidResponse
        }

        guard 200..<300 ~= httpResponse.statusCode else {
            throw AnalysisAPIError.requestFailed(
                statusCode: httpResponse.statusCode,
                message: Self.errorMessage(from: data)
            )
        }

        return try decoder.decode(SwingAIFeedback.self, from: data)
    }

    public func makeAnalyzeRequest(
        swingID: String,
        club: String? = nil,
        durationMs: Int?,
        trimStartMs: Int?,
        trimEndMs: Int?,
        videoEditState: VideoEditState? = nil,
        videoData: Data,
        filename: String,
        mimeType: String = "video/mp4"
    ) throws -> URLRequest {
        let endpoint = baseURL.appendingPathComponent("analyze-swing")
        var fields = ["swing_id": swingID]
        if let club, !club.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            fields["club"] = club
        }
        if let durationMs, durationMs > 0 {
            fields["client_duration_ms"] = String(durationMs)
        }
        if let trimStartMs {
            fields["trim_start_ms"] = String(max(0, trimStartMs))
        }
        if let trimEndMs {
            fields["trim_end_ms"] = String(max(0, trimEndMs))
        }
        if let videoEditState {
            let editState = videoEditState.normalized
            fields["rotation_degrees"] = String(editState.rotationDegrees)
            fields["mirror_horizontal"] = editState.isMirroredHorizontally ? "true" : "false"
        }

        let builder = MultipartFormDataBuilder()
        let body = builder.build(
            fields: fields,
            files: [
                MultipartFile(
                    fieldName: "video",
                    filename: filename,
                    mimeType: mimeType,
                    data: videoData
                ),
            ]
        )

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(builder.boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        request.timeoutInterval = 120
        return request
    }

    public func makeAIFeedbackRequest(
        swingID: String,
        club: String,
        analysisJSON: String,
        durationMs: Int?,
        trimStartMs: Int?,
        trimEndMs: Int?,
        videoEditState: VideoEditState? = nil,
        videoData: Data,
        filename: String,
        mimeType: String = "video/mp4"
    ) throws -> URLRequest {
        let endpoint = baseURL.appendingPathComponent("ai-analysis")
        var fields = [
            "swing_id": swingID,
            "club": club,
        ]
        if let durationMs, durationMs > 0 {
            fields["client_duration_ms"] = String(durationMs)
        }
        if let trimStartMs {
            fields["trim_start_ms"] = String(max(0, trimStartMs))
        }
        if let trimEndMs {
            fields["trim_end_ms"] = String(max(0, trimEndMs))
        }
        if let videoEditState {
            let editState = videoEditState.normalized
            fields["rotation_degrees"] = String(editState.rotationDegrees)
            fields["mirror_horizontal"] = editState.isMirroredHorizontally ? "true" : "false"
        }

        let builder = MultipartFormDataBuilder()
        let body = builder.build(
            fields: fields,
            files: [
                MultipartFile(
                    fieldName: "analysis_json",
                    filename: "\(swingID)-analysis.json",
                    mimeType: "application/json",
                    data: Data(analysisJSON.utf8)
                ),
                MultipartFile(
                    fieldName: "video",
                    filename: filename,
                    mimeType: mimeType,
                    data: videoData
                ),
            ]
        )

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(builder.boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        request.timeoutInterval = 180
        return request
    }

    private static func errorMessage(from data: Data) -> String {
        if
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let detail = json["detail"] as? String
        {
            return detail
        }

        return String(data: data, encoding: .utf8) ?? ""
    }

    private static func uploadFilename(swingID: String, videoURL: URL) -> String {
        let fileExtension = videoURL.pathExtension.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !fileExtension.isEmpty else {
            return "\(swingID).mov"
        }
        return "\(swingID).\(fileExtension)"
    }

    private static func mimeType(for videoURL: URL) -> String {
        switch videoURL.pathExtension.lowercased() {
        case "mov", "qt":
            return "video/quicktime"
        case "m4v", "mp4":
            return "video/mp4"
        default:
            return "application/octet-stream"
        }
    }
}
