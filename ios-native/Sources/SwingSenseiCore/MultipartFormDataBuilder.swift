import Foundation

public struct MultipartFile {
    public let fieldName: String
    public let filename: String
    public let mimeType: String
    public let data: Data

    public init(fieldName: String, filename: String, mimeType: String, data: Data) {
        self.fieldName = fieldName
        self.filename = filename
        self.mimeType = mimeType
        self.data = data
    }
}

public struct MultipartFormDataBuilder {
    public let boundary: String

    public init(boundary: String = "Boundary-\(UUID().uuidString)") {
        self.boundary = boundary
    }

    public func build(fields: [String: String], files: [MultipartFile]) -> Data {
        var body = Data()

        for key in fields.keys.sorted() {
            guard let value = fields[key] else { continue }
            body.appendString("--\(boundary)\r\n")
            body.appendString("Content-Disposition: form-data; name=\"\(key)\"\r\n\r\n")
            body.appendString("\(value)\r\n")
        }

        for file in files {
            body.appendString("--\(boundary)\r\n")
            body.appendString(
                "Content-Disposition: form-data; name=\"\(file.fieldName)\"; filename=\"\(file.filename)\"\r\n"
            )
            body.appendString("Content-Type: \(file.mimeType)\r\n\r\n")
            body.append(file.data)
            body.appendString("\r\n")
        }

        body.appendString("--\(boundary)--\r\n")
        return body
    }
}

private extension Data {
    mutating func appendString(_ string: String) {
        append(Data(string.utf8))
    }
}
