import CoreML
import CoreVideo
import Foundation
import Vision

public final class ClubHeadDetector {
    public struct Detection: Sendable {
        public let center: CGPoint
        public let boundingBox: CGRect
        public let confidence: Float
    }

    public enum DetectorError: Error {
        case modelLoadFailed
    }

    private let visionModel: VNCoreMLModel
    private let confidenceThreshold: Float

    public init(confidenceThreshold: Float = 0.25) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let model = try ClubHeadModel(configuration: config)
        self.visionModel = try VNCoreMLModel(for: model.model)
        self.confidenceThreshold = confidenceThreshold
    }

    public func detect(
        in pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation = .up
    ) throws -> Detection? {
        let request = VNCoreMLRequest(model: visionModel)
        request.imageCropAndScaleOption = .scaleFit
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation)
        try handler.perform([request])
        return Self.bestDetection(from: request.results, threshold: confidenceThreshold)
    }

    public func detect(
        in cgImage: CGImage,
        orientation: CGImagePropertyOrientation = .up
    ) throws -> Detection? {
        let request = VNCoreMLRequest(model: visionModel)
        request.imageCropAndScaleOption = .scaleFit
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation)
        try handler.perform([request])
        return Self.bestDetection(from: request.results, threshold: confidenceThreshold)
    }

    private static func bestDetection(
        from results: [VNObservation]?,
        threshold: Float
    ) -> Detection? {
        guard let observations = results as? [VNRecognizedObjectObservation] else { return nil }
        let qualified = observations.filter { $0.confidence >= threshold }
        guard let best = qualified.max(by: { $0.confidence < $1.confidence }) else { return nil }
        // Vision returns normalized boundingBox with origin at bottom-left. Convert to top-left.
        let visionBox = best.boundingBox
        let flippedBox = CGRect(
            x: visionBox.minX,
            y: 1.0 - visionBox.maxY,
            width: visionBox.width,
            height: visionBox.height
        )
        let center = CGPoint(x: flippedBox.midX, y: flippedBox.midY)
        return Detection(center: center, boundingBox: flippedBox, confidence: best.confidence)
    }
}
