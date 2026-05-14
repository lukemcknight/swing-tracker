import AVFoundation
import SwiftUI

struct TransformedPlayerView: View {
    let player: AVPlayer
    let editState: VideoEditState

    var body: some View {
        GeometryReader { geometry in
            let size = geometry.size
            let currentEditState = editState.normalized
            let rotated = currentEditState.rotatesDimensions

            PlayerLayerView(player: player)
                .frame(
                    width: rotated ? size.height : size.width,
                    height: rotated ? size.width : size.height
                )
                .rotationEffect(.degrees(Double(currentEditState.rotationDegrees)))
                .scaleEffect(x: currentEditState.isMirroredHorizontally ? -1 : 1, y: 1)
                .position(x: size.width / 2, y: size.height / 2)
        }
        .clipped()
    }
}

struct PlayerLayerView: UIViewRepresentable {
    let player: AVPlayer

    func makeUIView(context: Context) -> PlayerUIView {
        let view = PlayerUIView()
        view.playerLayer.player = player
        view.playerLayer.videoGravity = .resizeAspect
        return view
    }

    func updateUIView(_ uiView: PlayerUIView, context: Context) {
        uiView.playerLayer.player = player
    }
}

final class PlayerUIView: UIView {
    override static var layerClass: AnyClass {
        AVPlayerLayer.self
    }

    var playerLayer: AVPlayerLayer {
        layer as! AVPlayerLayer
    }
}
