import AVFoundation
import CoreGraphics
import Foundation

final class VideoFrameSampler {
    private let generator: AVAssetImageGenerator

    init(url: URL) {
        let asset = AVURLAsset(url: url)
        let g = AVAssetImageGenerator(asset: asset)
        g.appliesPreferredTrackTransform = true
        // Allow ~half a frame at 30fps; exact-zero tolerance fails on HEVC/Simulator.
        g.requestedTimeToleranceBefore = CMTime(value: 16, timescale: 1000)
        g.requestedTimeToleranceAfter = CMTime(value: 16, timescale: 1000)
        self.generator = g
    }

    func image(atMs timeMs: Int) async throws -> CGImage {
        let time = CMTime(value: CMTimeValue(timeMs), timescale: 1000)
        let (image, _) = try await generator.image(at: time)
        return image
    }
}
