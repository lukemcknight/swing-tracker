import Foundation

public enum SwingRepositoryError: Error, LocalizedError {
    case missingVideo(URL)

    public var errorDescription: String? {
        switch self {
        case let .missingVideo(url):
            "Could not find video at \(url.path)."
        }
    }
}

public final class SwingRepository {
    public let rootDirectory: URL
    public let videosDirectory: URL
    public let metadataURL: URL

    private let fileManager: FileManager
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    public init(
        rootDirectory: URL? = nil,
        fileManager: FileManager = .default
    ) {
        self.fileManager = fileManager

        let baseDirectory = rootDirectory ?? fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("SwingSensei", isDirectory: true)
        self.rootDirectory = baseDirectory
        self.videosDirectory = baseDirectory.appendingPathComponent("videos", isDirectory: true)
        self.metadataURL = baseDirectory.appendingPathComponent("swings.json")

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        self.encoder = encoder

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        self.decoder = decoder
    }

    public func load() throws -> [SwingRecord] {
        try ensureDirectoriesExist()
        guard fileManager.fileExists(atPath: metadataURL.path) else {
            return []
        }

        let data = try Data(contentsOf: metadataURL)
        return try decoder.decode([SwingRecord].self, from: data)
            .sorted { $0.createdAt > $1.createdAt }
    }

    public func save(_ swings: [SwingRecord]) throws {
        try ensureDirectoriesExist()
        let data = try encoder.encode(swings.sorted { $0.createdAt > $1.createdAt })
        try data.write(to: metadataURL, options: [.atomic])
    }

    public func copyVideo(
        from sourceURL: URL,
        source: SwingSource,
        durationMs: Int?,
        orientation: SwingOrientation = .unknown,
        id: String = "swing-\(UUID().uuidString)",
        createdAt: Date = Date()
    ) throws -> SwingRecord {
        try ensureDirectoriesExist()
        guard fileManager.fileExists(atPath: sourceURL.path) else {
            throw SwingRepositoryError.missingVideo(sourceURL)
        }

        let destination = videoDestinationURL(for: id, sourceURL: sourceURL)
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
        try fileManager.copyItem(at: sourceURL, to: destination)

        return SwingRecord(
            id: id,
            videoURL: destination,
            createdAt: createdAt,
            source: source,
            orientation: orientation,
            durationMs: durationMs,
            trimStartMs: 0,
            trimEndMs: durationMs,
            status: .trimming
        )
    }

    public func upserting(_ swing: SwingRecord, into swings: [SwingRecord]) -> [SwingRecord] {
        [swing] + swings.filter { $0.id != swing.id }
    }

    public func videoDestinationURL(for id: String, sourceURL: URL) -> URL {
        let fileExtension = sourceURL.pathExtension.isEmpty ? "mov" : sourceURL.pathExtension
        return videosDirectory
            .appendingPathComponent(id)
            .appendingPathExtension(fileExtension)
    }

    private func ensureDirectoriesExist() throws {
        try fileManager.createDirectory(at: rootDirectory, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: videosDirectory, withIntermediateDirectories: true)
    }
}
