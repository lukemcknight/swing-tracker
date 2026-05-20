import Foundation

public extension SwingAnalysis {
    func normalizingDebugURLs(baseURL: URL) -> SwingAnalysis {
        let normalizedDebug = debug.map { debug in
            SwingAnalysisDebug(
                contactSheetError: debug.contactSheetError,
                contactSheetPath: debug.contactSheetPath,
                contactSheetUrl: Self.normalizedURLString(debug.contactSheetUrl, baseURL: baseURL),
                impactFilmstripError: debug.impactFilmstripError,
                impactFilmstripPath: debug.impactFilmstripPath,
                impactFilmstripUrl: Self.normalizedURLString(debug.impactFilmstripUrl, baseURL: baseURL),
                impactWindowTimesMs: debug.impactWindowTimesMs,
                positionTimesMs: debug.positionTimesMs,
                analysisQuality: debug.analysisQuality,
                phaseConfidence: debug.phaseConfidence
            )
        }

        return SwingAnalysis(
            id: id,
            videoUri: videoUri,
            fps: fps,
            durationMs: durationMs,
            analysisVersion: analysisVersion,
            debug: normalizedDebug,
            videoWidth: videoWidth,
            videoHeight: videoHeight,
            trimStartMs: trimStartMs,
            trimEndMs: trimEndMs,
            frames: frames,
            positions: positions,
            metrics: metrics,
            feedback: feedback,
            swingThought: swingThought,
            analysisQuality: analysisQuality,
            phaseConfidence: phaseConfidence,
            shotEstimate: shotEstimate,
            rawSenseiScore: rawSenseiScore,
            senseiScore: senseiScore
        )
    }

    private static func normalizedURLString(_ value: String?, baseURL: URL) -> String? {
        guard let value, !value.isEmpty else { return value }
        if URL(string: value)?.scheme != nil {
            return value
        }
        return URL(string: value, relativeTo: baseURL)?.absoluteURL.absoluteString ?? value
    }
}
